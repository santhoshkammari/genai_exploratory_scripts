import random
import re
import sys
import os
import json
import logging
import time
import os

from langchain_ollama import ChatOllama

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QVBoxLayout, QTextEdit, QDesktopWidget, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QFont, QTextCursor
from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
import threading
# from ollama_fix import CustomChatOllama
from pyopengenai import CustomChatOllama
# Register QTextCursor for use in signals
from PyQt5.QtCore import QMetaType

from universe_prompt import UP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QMetaType.type("QTextCursor")

class StreamHandler(QObject):
    new_token_signal = pyqtSignal(str)
    
    def __init__(self, response_callback):
        super().__init__()
        self.response_callback = response_callback
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.new_token_signal.emit(token)
        self.full_response += token

    def on_llm_end(self, response, **kwargs):
        self.response_callback(self.full_response)

DEFAULT_MODELS: list = ["qwen2:1.5b", "gemma2:2b"]

class SpotlightLLM(QWidget):
    def __init__(self, models:list = None, execution_mode="Local"):
        super().__init__()
        self.models = models or DEFAULT_MODELS
        self.execution_mode = execution_mode  # Default to Local execution
        self.current_model = self.models[0]
        self.initUI()
        self.setup_data_directory()
        self.session_interactions = []
        self.response_complete = threading.Event()
        
        # Set up logging
        logging.basicConfig(filename='spotlight_llm.log', level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Create a horizontal layout for search bar, model selection, and execution mode
        search_layout = QHBoxLayout()

        self.search_bar = QLineEdit(self)
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 20px;
                padding: 10px;
                color: white;
                font-size: 18px;
            }
        """)
        self.search_bar.returnPressed.connect(self.on_submit)
        search_layout.addWidget(self.search_bar, 5)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.models)
        self.model_selector.setStyleSheet(self.get_combobox_style())
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        search_layout.addWidget(self.model_selector, 2)

        self.execution_selector = QComboBox(self)
        self.execution_selector.addItems(["Local", "GPU"])
        self.execution_selector.setStyleSheet(self.get_combobox_style())
        self.execution_selector.currentTextChanged.connect(self.on_execution_mode_change)
        search_layout.addWidget(self.execution_selector, 2)

        layout.addLayout(search_layout)

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 10px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
        """)
        self.result_area.hide()
        layout.addWidget(self.result_area)

        self.setLayout(layout)

        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        screen_size = QDesktopWidget().screenGeometry(screen)
        window_width = 750
        window_height = 60
        x = (screen_size.width() - window_width) // 2
        y = screen_size.height() // 4
        self.setGeometry(x, y, window_width, window_height)

    def get_combobox_style(self):
        return """
            QComboBox {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 20px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                selection-background-color: rgba(80, 80, 80, 200);
                color: white;
            }
        """

    def setup_data_directory(self):
        self.data_dir = os.path.join(os.path.expanduser("~"), "spotlight_llm_data")
        os.makedirs(self.data_dir, exist_ok=True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(30, 30, 30, 200))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

    def on_model_change(self, model):
        self.current_model = model
        logging.info(f"Model changed to: {model}")

    def on_execution_mode_change(self, mode):
        self.execution_mode = mode
        logging.info(f"Execution mode changed to: {mode}")

    def on_submit(self):
        query = self.search_bar.text()
        if query:
            self.result_area.clear()
            self.result_area.show()
            self.animate_expand()
            self.current_query = query
            self.current_timestamp = datetime.now().isoformat()
            self.response_complete.clear()
            threading.Thread(target=self.get_response, args=(query,), daemon=True).start()

    def update_result_area(self, token):
        cursor = self.result_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.result_area.setTextCursor(cursor)
        self.result_area.ensureCursorVisible()

    def add_universe_prompt(self,prompt):
        return UP + f"\n User Question : {prompt}"

    def format_results_multi_agent(self, results):
        output = []

        # Add the final answer
        output.append("Final Answer:\n")
        output.append(results['final_answer'])
        output.append("")

        # Add subqueries and their results
        output.append("Subqueries and WebAnalysis:\n")
        for i, subquery in enumerate(results['subqueries']):
            output.append(f"{i + 1}. Subquery: {subquery}\n")

            # Get the corresponding result
            result = results['subquery_results'][i]

            # output.append("   Web Result:")
            # output.append(f"   {result['web_result']}")

            output.append("Relevant Information:")
            output.append(f"{result['relevant_info']}")
            output.append("")

        return "\n".join(output)

    def filter_prompts(self, prompt):
        # Add your prompt filtering logic here
        filtered_prompt = " ".join(prompt.split()[:-1])

        prompt = re.split("User Question : ",filtered_prompt)[-1]
        return prompt
    def get_response(self, prompt):
        prompt = self.add_universe_prompt(prompt)
        logging.debug(f"Starting response generation in {self.execution_mode} mode")
        stream_handler = StreamHandler(self.save_interaction)
        stream_handler.new_token_signal.connect(self.update_result_area)

        try:
            if self.execution_mode == "Local":
                llm = Ollama(model=self.current_model)
            elif self.execution_mode == "GPU":
                llm = CustomChatOllama(model=self.current_model, base_url="http://192.168.162.49:8888")
            else:
                raise ValueError('executoi mode wrong')

            if prompt.lower().split()[-1] in ["google"]:
                from pyopengenai.researcher_ai.main.researcher import AiResearcher
                from pyopengenai.researcher_ai.main.llm_service.base import LLMPrompt

                researcher = AiResearcher()
                question = self.filter_prompts(prompt)
                print(f"Question: {question}")
                content,urls = researcher.get_query_content(query=question,
                                                       max_urls=2,max_articles=2,
                                                       return_urls = True)
                context = "\n".join(content)
                logger.info(f"URLS: {urls}")
                runner = LLMPrompt(template="""
                You are Expert summarizer, given user question and context understand and provide using context only.
                Answer User question: {question}
                Context: {context}
                """,model=self.current_model,device=self.execution_mode)
                for c in runner.stream(question=question,context=context):
                    stream_handler.on_llm_new_token(c)
                urls = ["\n\nUrls:"]+urls
                for url in urls:
                    stream_handler.on_llm_new_token(f'{url}\n')

            elif prompt.lower().split()[-1] in ["low","medium","high"]:
                from pyopengenai.researcher_ai import MultiAgentQueryOrchestrator
                processor = MultiAgentQueryOrchestrator(
                    response_level=prompt.lower().split()[-1],
                    device=self.execution_mode
                )
                filtered_prompt = self.filter_prompts(prompt)
                processor.process_query(filtered_prompt)
                results = self.format_results_multi_agent(processor.results)
                delays = [0.0015, 0.0005, 0.0025]
                for chunk in results:  # Split the results into words
                    time.sleep(random.choice(delays))# Add a small delay between each word
                    stream_handler.on_llm_new_token(chunk)

            else:
                # GPU mode
                # llm = ChatOllama(model="qwen2:0.5b")
                for chunk in llm.stream(prompt):
                    stream_handler.on_llm_new_token(chunk.content)

            stream_handler.on_llm_end(None)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QApplication.instance().postEvent(self.result_area, QTextCursor(self.result_area.document()))
            self.result_area.setPlainText(error_message)
            self.save_interaction(error_message)
            logging.exception(f"Error in {self.execution_mode} mode")
        finally:
            if self.execution_mode == "GPU":
                # Add any necessary cleanup for GPU resources
                pass
            logging.debug(f"Finished response generation in {self.execution_mode} mode")
            self.response_complete.set()

    def save_interaction(self, response):
        interaction_data = {
            "timestamp": self.current_timestamp,
            "model": self.current_model,
            "execution_mode": self.execution_mode,
            "query": self.current_query,
            "response": response
        }
        self.session_interactions.append(interaction_data)
        logging.info(f"Interaction saved: {self.current_timestamp}")

    def animate_expand(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(self.geometry())
        new_height = 400
        new_geometry = QRect(self.x(), self.y(), self.width(), new_height)
        self.animation.setEndValue(new_geometry)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.save_session()
            self.close()

    def save_session(self):
        if self.session_interactions:
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(self.session_interactions, f, indent=2)
            logging.info(f"Session saved: {filepath}")

    def closeEvent(self, event):
        if self.execution_mode == "GPU":
            self.response_complete.wait(timeout=10)  # Wait up to 10 seconds for response to complete
        self.save_session()
        logging.info("Application closing")
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpotlightLLM(
        # execution_mode="GPU",
        # models=["qwen2:7b-instruct"]
        models = ['qwen2:0.5b']
    )
    ex.show()
    sys.exit(app.exec_())
