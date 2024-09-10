from io import BytesIO
import torch
import torch.package
import torchaudio
from quart import Quart, request, jsonify, send_file
from pydub import AudioSegment
from quart_cors import cors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ru_model_path = "ru.pt"
en_model_path = "en.pt"
tts_models = "tts_models"
ru_speaker = "baya"
en_speaker = "en_100"
sample_rate = 48000
app = Quart(__name__)
app = cors(app, allow_origin=["https://localhost:5068", "https://localhost:7089"])


@app.route('/voiceover_of_text', methods=['POST'])
async def voiceover_of_text():
    language = request.args.get("language")
    app.logger.info("language: " + language)
    data = await request.get_json()
    text = data["text"]
    app.logger.info("text: " + text)
    if not text:
        return jsonify({"error": "No text provided"}), 400
    speaker = ""
    if "ru" in language:
        model = torch.package.PackageImporter(ru_model_path).load_pickle(tts_models, "model")
        speaker = ru_speaker
    elif "en" in language:
        model = torch.package.PackageImporter(en_model_path).load_pickle(tts_models, "model")
        speaker = en_speaker
    else:
        return jsonify({"error": "No provided language"}), 400
    model.to(device)
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
    buffer = BytesIO()
    torchaudio.save(uri=buffer, src=audio.unsqueeze(0), sample_rate=sample_rate, format="wav")
    buffer.seek(0)

    audio_segment = AudioSegment.from_wav(buffer)
    buffer_mp3 = BytesIO()
    audio_segment.export(buffer_mp3, format="mp3", bitrate="192k")
    buffer_mp3.seek(0)

    return await send_file(buffer_mp3, mimetype="audio/mp3", as_attachment=True, attachment_filename="voiceover.mp3")


if __name__ == '__main__':
    app.run(debug=True, port=11111)
