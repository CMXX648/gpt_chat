import os
from wakeup import WakeupHandler
from audio_player import play_audio, play_wakeup, play_waiting
from audio_recorder import record_audio
from paddle_service_api import NLPService
from gpt_service_api import qwen_chat
from prompt import PROMPT
from utils import get_json
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Pipeline():
    def __init__(self, settings):
        self.wakeup_handler = WakeupHandler(settings['porcupine'])
        self.nlp_service = NLPService(settings['nlp_service'])

    def run(self, model_chat_func, questionnaire_text, initial_prompt, output_file):
        logger.info('running...')
        prompt = f"{initial_prompt} {questionnaire_text}"
        while True:
            keyword = self.wakeup_handler.run_detect_wakeup_word()
            if not keyword:
                continue
            # 1. Wait for the wake word
            logger.info('等待唤醒词，倾听中...')
            play_wakeup()

            # 2. Record your voice
            temp_audio_path = 'temp.wav'
            record_audio(temp_audio_path, 4)  # 4秒后结束倾听

            # 3. Call Automatic Speech Recognition api
            query_text = self.nlp_service.get_asr_result(temp_audio_path)

            # 如果模型未获得语音转录结果
            if not query_text:
                logger.info("模型未听到任何答复，请重试")
            logger.info('query text: {}'.format(query_text))

            # 4. Wait for GPT reply
            play_waiting()
            response_text = model_chat_func(prompt, query_text)
            logger.info('Question: {}'.format(response_text))
            temp_response_audio_path = 'temp_res.wav'
            logger.info('gen speech...')

            # 5. Call Text to speech api
            self.nlp_service.get_tts_result(
                response_text, temp_response_audio_path)
            logger.info('play response...')

            # 6. Play the speech
            play_audio(temp_response_audio_path)
            logger.info('倾听中...')

            # 7. Clean up temporary files
            os.remove(temp_audio_path)
            os.remove(temp_response_audio_path)


if __name__ == '__main__':
    file_path = 'Q.txt'
    output_file = 'final.txt'
    settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Gpt_Chat/resource/settings.json')
    initial_prompt = f"按照以下要求来实现本次对话。\n \"\"\"{PROMPT}\"\"\"。"

    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        questionnaire_text = f"这是问卷内容：\n \"\"\"{file_content}\"\"\""

    settings = get_json(settings_file)
    pipeline = Pipeline(settings)
    pipeline.run(qwen_chat, questionnaire_text, initial_prompt, output_file)

    # 在最终生成建议之前应用反思机制
    with open(output_file, 'r', encoding='utf-8') as file:
        dialogue_history = file.read()

    final_output = perform_reflection(dialogue_history)
    print(final_output)

