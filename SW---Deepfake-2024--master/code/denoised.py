import soundfile as sf
import torch
from pathlib import Path
from df.enhance import enhance, init_df, load_audio, save_audio

# DeepFilterNet을 사용한 노이즈 제거 함수
def denoise_audio_from_ogg(input_path, output_path):
    data, samplerate = sf.read(input_path)
    data_tensor = torch.from_numpy(data.astype('float32')).unsqueeze(0)  
    model, df_state, _ = init_df()  
    enhanced_audio = enhance(model, df_state, data_tensor)
    enhanced_audio = enhanced_audio.squeeze(0).numpy()  
    save_audio(output_path, enhanced_audio, samplerate)
    print(f"Saved denoised audio to: {output_path}")

if __name__ == "__main__":
    # 경로 설정
    ogg_dir = Path("./test")  # OGG 파일이 있는 디렉토리
    output_dir = Path("./test_denoised")  # 소음 제거된 파일을 저장할 디렉토리
    output_dir.mkdir(exist_ok=True)  # 출력 디렉토리가 없으면 생성

    # 디렉토리 내의 모든 OGG 파일을 순회
    for ogg_file in ogg_dir.glob("*.ogg"):
        output_file = output_dir / (ogg_file.stem + "_denoised.wav")
        denoise_audio_from_ogg(ogg_file, output_file)
