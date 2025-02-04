import os
import tempfile
import logging
from functools import wraps

from flask import Flask, request, send_file, jsonify
import soundfile as sf
import torch
import numpy as np
from pydub import AudioSegment


from df import enhance, init_df

app = Flask(__name__)
app.config['DEBUG'] = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SECRET_TOKEN = "BacBon"


logger.info("Initializing DeepFilterNet model...")
model, df_state, _ = init_df()
logger.info("Model loaded successfully.")


UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav(input_path):
    """
    Convert any supported audio file to a 48 kHz, mono WAV file.
    Returns the path to the converted file.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(48000).set_channels(1)
        converted_path = os.path.splitext(input_path)[0] + "_converted.wav"
        audio.export(converted_path, format="wav")
        logger.info("File converted to 48kHz mono WAV: %s", converted_path)
        return converted_path
    except Exception as e:
        logger.exception("Error converting audio file:")
        raise e


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", None)
        if not auth_header:
            return jsonify({'error': 'Authorization header is missing.'}), 401

        try:
            token_type, token = auth_header.split()
        except ValueError:
            return jsonify({'error': 'Invalid Authorization header format.'}), 401

        if token_type.lower() != "bearer":
            return jsonify({'error': 'Invalid token type. Bearer token required.'}), 401

        if token != SECRET_TOKEN:
            return jsonify({'error': 'Invalid or missing token.'}), 401

        return f(*args, **kwargs)
    return decorated


@app.route('/api/clean', methods=['POST'])
@token_required
def clean_audio():

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    if file and allowed_file(file.filename):

        orig_input_path = os.path.join(
            app.config['UPLOAD_FOLDER'], file.filename)
        file.save(orig_input_path)
        logger.info("Uploaded file saved to: %s", orig_input_path)

        try:

            converted_path = convert_to_wav(orig_input_path)

            noisy_audio, sr = sf.read(converted_path)
            logger.info("Converted file sample rate: %s", sr)
            if sr != 48000:
                return jsonify({'error': 'Converted audio sample rate is not 48 kHz.'}), 400

            if noisy_audio.ndim == 2:
                logger.info(
                    "Converted file is stereo; converting to mono by averaging channels.")
                noisy_audio = np.mean(noisy_audio, axis=1)
            else:
                logger.info("Converted file is mono.")

            noisy_audio_tensor = torch.tensor(noisy_audio, dtype=torch.float32)
            if noisy_audio_tensor.ndim == 1:
                noisy_audio_tensor = noisy_audio_tensor.unsqueeze(0)
            logger.info("Input tensor shape: %s", noisy_audio_tensor.shape)

            enhanced_audio_tensor = enhance(
                model, df_state, noisy_audio_tensor)
            logger.info("Enhancement complete. Output tensor shape: %s",
                        enhanced_audio_tensor.shape)

            enhanced_audio = (
                enhanced_audio_tensor.cpu().numpy()
                if isinstance(enhanced_audio_tensor, torch.Tensor)
                else enhanced_audio_tensor
            )

            if enhanced_audio.ndim == 2 and enhanced_audio.shape[0] == 1:
                enhanced_audio = np.squeeze(enhanced_audio, axis=0)
                logger.info("Squeezed enhanced audio shape: %s",
                            enhanced_audio.shape)

            output_filename = "cleaned_" + file.filename.split('.')[0] + ".wav"
            output_path = os.path.join(
                app.config['UPLOAD_FOLDER'], output_filename)
            sf.write(output_path, enhanced_audio, sr)
            logger.info("Enhanced file saved at: %s", output_path)

            return send_file(output_path, as_attachment=True, mimetype="audio/wav")

        except Exception as e:
            logger.exception("Error during processing:")
            return jsonify({'error': f'An error occurred during processing: {e}'}), 500
    else:
        return jsonify({'error': 'File type not allowed. Supported formats: ' + ", ".join(ALLOWED_EXTENSIONS)}), 400


if __name__ == '__main__':
    app.run(debug=True)
