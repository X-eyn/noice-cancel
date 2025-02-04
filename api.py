import os
import tempfile
import logging
from functools import wraps

from flask import Flask, request, send_file, jsonify, url_for
import soundfile as sf
import torch
import numpy as np
from pydub import AudioSegment  # For audio conversion

# Import DeepFilterNet functionality
from df import enhance, init_df

app = Flask(__name__)
app.config['DEBUG'] = True  # Set to False in production

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your secret token here (in production, store securely)
SECRET_TOKEN = "BacBon"

# Initialize DeepFilterNet model
logger.info("Initializing DeepFilterNet model...")
model, df_state, _ = init_df()
logger.info("Model loaded successfully.")

# Use a temporary folder for file operations
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for incoming audio files
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
    # Check for file in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file to a temporary location
        orig_input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(orig_input_path)
        logger.info("Uploaded file saved to: %s", orig_input_path)
        
        try:
            # Convert the file to 48kHz mono WAV if needed
            converted_path = convert_to_wav(orig_input_path)
            
            # Read the converted audio file
            noisy_audio, sr = sf.read(converted_path)
            logger.info("Converted file sample rate: %s", sr)
            if sr != 48000:
                return jsonify({'error': 'Converted audio sample rate is not 48 kHz.'}), 400
            
            # If stereo, average channels (should be mono after conversion, but double-check)
            if noisy_audio.ndim == 2:
                logger.info("Converted file is stereo; converting to mono by averaging channels.")
                noisy_audio = np.mean(noisy_audio, axis=1)
            else:
                logger.info("Converted file is mono.")
            
            # Convert to PyTorch tensor and ensure 2D shape: [channels, samples]
            noisy_audio_tensor = torch.tensor(noisy_audio, dtype=torch.float32)
            if noisy_audio_tensor.ndim == 1:
                noisy_audio_tensor = noisy_audio_tensor.unsqueeze(0)
            logger.info("Input tensor shape: %s", noisy_audio_tensor.shape)
            
            # Clean the audio with DeepFilterNet
            enhanced_audio_tensor = enhance(model, df_state, noisy_audio_tensor)
            logger.info("Enhancement complete. Output tensor shape: %s", enhanced_audio_tensor.shape)
            
            # Convert output tensor back to numpy array
            enhanced_audio = (
                enhanced_audio_tensor.cpu().numpy()
                if isinstance(enhanced_audio_tensor, torch.Tensor)
                else enhanced_audio_tensor
            )
            
            # If enhanced audio is mono but still has an extra channel dimension, squeeze it
            if enhanced_audio.ndim == 2 and enhanced_audio.shape[0] == 1:
                enhanced_audio = np.squeeze(enhanced_audio, axis=0)
                logger.info("Squeezed enhanced audio shape: %s", enhanced_audio.shape)
            
            # Save the enhanced audio file
            output_filename = "cleaned_" + file.filename.split('.')[0] + ".wav"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            sf.write(output_path, enhanced_audio, sr)
            logger.info("Enhanced file saved at: %s", output_path)
            
            # Construct a temporary download URL. In a production scenario, you might generate a time-limited link.
            download_url = request.host_url.rstrip('/') + url_for('download_file', filename=output_filename)
            return jsonify({"download_url": download_url}), 200
        
        except Exception as e:
            logger.exception("Error during processing:")
            return jsonify({'error': f'An error occurred during processing: {e}'}), 500
    else:
        return jsonify({'error': 'File type not allowed. Supported formats: ' + ", ".join(ALLOWED_EXTENSIONS)}), 400

@app.route('/api/download/<filename>', methods=['GET'])
@token_required
def download_file(filename):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True, mimetype="audio/wav")
    else:
        return jsonify({'error': 'File not found.'}), 404

if __name__ == '__main__':
    app.run(debug=True)
