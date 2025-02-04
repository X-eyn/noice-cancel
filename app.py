import os
import tempfile
import logging

from flask import Flask, request, render_template, send_file, redirect, flash
import soundfile as sf
import torch
import numpy as np
from pydub import AudioSegment  # Used for audio conversion

# Import DeepFilterNet functionality
from df import enhance, init_df

# Create the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key in production

# Set up logging to console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the DeepFilterNet model (this may take a moment on the first run)
logger.info("Initializing DeepFilterNet model...")
model, df_state, _ = init_df()
logger.info("Model loaded successfully.")

# Set the upload folder (using the system temporary directory)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extension(s) for uploaded audio files.
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path):
    """
    Convert any supported audio file to a 48 kHz, mono WAV file.
    Returns the path to the converted file.
    """
    try:
        # Load the file using pydub
        audio = AudioSegment.from_file(input_path)
        # Set the frame rate to 48000 and channels to mono
        audio = audio.set_frame_rate(48000).set_channels(1)
        # Save the converted file as a WAV
        converted_path = os.path.splitext(input_path)[0] + "_converted.wav"
        audio.export(converted_path, format="wav")
        logger.info("File converted to 48kHz mono WAV: %s", converted_path)
        return converted_path
    except Exception as e:
        logger.exception("Error converting audio file:")
        raise e

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is included in the request
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the uploaded file
            orig_input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(orig_input_path)
            logger.info("Uploaded file saved to: %s", orig_input_path)
            
            try:
                # Convert the file to 48kHz mono WAV if needed
                converted_path = convert_to_wav(orig_input_path)
                
                # Read the converted audio file
                noisy_audio, sr = sf.read(converted_path)
                logger.info("Converted file sample rate: %s", sr)
                
                # (Optional) Verify sample rate is 48000
                if sr != 48000:
                    flash("Converted audio sample rate is not 48 kHz.")
                    return redirect(request.url)
                
                # At this point, the file should be mono; check dimensionality
                if noisy_audio.ndim == 2:
                    logger.info("Converted file is stereo; converting to mono by averaging channels.")
                    noisy_audio = np.mean(noisy_audio, axis=1)
                else:
                    logger.info("Converted file is mono.")
                
                # Convert the numpy array to a PyTorch tensor (float32 is required)
                noisy_audio_tensor = torch.tensor(noisy_audio, dtype=torch.float32)
                
                # Ensure the tensor has two dimensions: [channels, samples]
                if noisy_audio_tensor.ndim == 1:
                    noisy_audio_tensor = noisy_audio_tensor.unsqueeze(0)
                logger.info("Input converted to tensor with shape: %s", noisy_audio_tensor.shape)
                
                # Process the tensor with DeepFilterNet
                enhanced_audio_tensor = enhance(model, df_state, noisy_audio_tensor)
                logger.info("Enhancement complete. Output tensor shape: %s", enhanced_audio_tensor.shape)
                
                # Convert output tensor back to numpy array if needed
                enhanced_audio = (
                    enhanced_audio_tensor.cpu().numpy()
                    if isinstance(enhanced_audio_tensor, torch.Tensor)
                    else enhanced_audio_tensor
                )
                
                # If enhanced audio is mono but has an extra singleton dimension, squeeze it
                if enhanced_audio.ndim == 2 and enhanced_audio.shape[0] == 1:
                    enhanced_audio = np.squeeze(enhanced_audio, axis=0)
                    logger.info("Squeezed enhanced audio shape: %s", enhanced_audio.shape)
                
                # Save the enhanced audio to a temporary file
                output_filename = "cleaned_" + file.filename.split('.')[0] + ".wav"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                sf.write(output_path, enhanced_audio, sr)
                logger.info("Enhanced file saved at: %s", output_path)
                
                # Verify the output file exists and send it to the user
                if os.path.exists(output_path):
                    return send_file(output_path, as_attachment=True, mimetype="audio/wav")
                else:
                    flash("Processed file not found.")
                    logger.error("Processed file was not found at: %s", output_path)
                    return redirect(request.url)
            except Exception as e:
                flash(f"An error occurred during processing: {e}")
                logger.exception("Error during processing:")
                return redirect(request.url)
        else:
            flash("Allowed file types are: " + ", ".join(ALLOWED_EXTENSIONS))
            return redirect(request.url)
    return render_template('upload.html')

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True)
