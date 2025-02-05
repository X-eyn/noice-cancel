# DeepFilterNet Audio Enhancement API

This project provides a Flask-based API that processes audio files using the DeepFilterNet framework for noise reduction and enhancement. It supports multiple common audio formats by automatically converting them to a 48 kHz, mono WAV format before processing. The API returns a JSON response containing a temporary download URL for the enhanced audio file. Additionally, critical endpoints are secured with bearer token authentication.

## Features

- **Multi-Format Support:**  
  Accepts WAV, MP3, M4A, OGG, FLAC, etc., and converts them to 48 kHz, mono WAV.
  
- **Audio Enhancement:**  
  Uses DeepFilterNet (default: DeepFilterNet3) to perform noise reduction.
  
- **JSON Response:**  
  Returns a JSON response with a temporary download URL instead of directly sending the file.
  
- **Authentication:**  
  Protects the cleaning endpoint with bearer token authentication. (The download endpoint can be secured or made public.)

## Requirements

- **Python 3.8+**
- **FFmpeg:**  
  Required for audio conversion via `pydub`. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure the `ffmpeg` executable is in your system PATH.
- **Python Packages:**  
  Install the required packages using the provided `requirements.txt` file.


## Setup Instructions

1. **Clone or Download the Project Files:**

   ```bash
   git clone https://github.com/yourusername/your-project-repo.git
   cd your-project-repo
   ```

2. **Create and Activate a Virtual Environment:**

   - On Windows:

     python -m venv venv
     venv\Scripts\activate.bat

   - On macOS/Linux:
     
     python3 -m venv venv
     source venv/bin/activate
     

3. **Install Dependencies:**

   
   pip install -r requirements.txt
   

4. **Configure the Project:**

   - Ensure that FFmpeg is installed and its binary is in your PATH.
   - Open `api.py` and set your secret token by modifying the following line:
     
     SECRET_TOKEN = "YOUR_SECRET_TOKEN"
     
     Replace `"YOUR_SECRET_TOKEN"` with your desired token.

## Running the API Server

To start the API server, run:


python api.py


The server will start locally at `http://127.0.0.1:5000`.

## API Endpoints

### 1. `/api/clean` (POST)

- **Description:**  
  Accepts an audio file for noise reduction and returns a JSON response containing a temporary download URL for the cleaned file.
  
- **Authentication:**  
  Requires a bearer token in the `Authorization` header (format: `Bearer YOUR_SECRET_TOKEN`).

- **Request:**
  - **Parameter:**  
    `file` – The audio file (multipart/form-data).

- **Response (Success):**  
  ```json
  {
    "download_url": "http://127.0.0.1:5000/api/download/cleaned_filename.wav"
  }
  ```
  
- **Example cURL Command:**

  ```bash
  curl.exe -X POST -H "Authorization: Bearer YOUR_SECRET_TOKEN" -F "file=@C:\Users\Admin\Downloads\noise\your_audio_file.mp3" http://127.0.0.1:5000/api/clean
  ```

### 2. `/api/download/<filename>` (GET)

- **Description:**  
  Returns the cleaned audio file for download.
  
- **Authentication:**  
  By default, this endpoint is protected by the same bearer token. (To make it public, remove the `@token_required` decorator in the code.)

- **Response:**  
  The cleaned audio file is returned as an attachment with the MIME type `audio/wav`.

- **Example cURL Command:**


  curl.exe -X GET -H "Authorization: Bearer YOUR_SECRET_TOKEN" http://127.0.0.1:5000/api/download/cleaned_your_audio_file.wav --output final_cleaned_audio.wav
  

## Workflow Overview

1. **Client Request:**  
   A client sends a POST request to `/api/clean` with an audio file and the correct bearer token.

2. **Authentication & Validation:**  
   The API verifies the token and checks that a valid file was uploaded.

3. **Audio Conversion:**  
   The file is converted (if necessary) to a 48 kHz, mono WAV format using `pydub`.

4. **Audio Enhancement:**  
   The converted file is read, preprocessed, and processed using the DeepFilterNet model (defaulting to DeepFilterNet3).

5. **Response Construction:**  
   The enhanced audio file is saved to a temporary folder, and a JSON response with a temporary download URL is returned.

6. **File Download:**  
   Clients can retrieve the enhanced file using the provided download URL.

## Security Notes

- **Bearer Token:**  
  Only clients with the correct bearer token can access the `/api/clean` endpoint.  
- **Download Endpoint:**  
  The download endpoint is also secured by default but can be made public if desired.
- **Production Considerations:**  
  For production, use HTTPS, secure your token (e.g., via environment variables), and consider implementing rate limiting and cleanup of temporary files.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- Other open-source libraries and tools used in this project.

---