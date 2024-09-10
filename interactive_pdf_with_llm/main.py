import sys

import markdown
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QTextEdit, QScrollArea, QFrame, QSplitter, QComboBox, QLabel,
                             QPlainTextEdit, QListWidget)
from PyQt5.QtGui import QPainter, QColor, QImage, QPen, QFont, QPalette
from PyQt5.QtCore import Qt, QRectF, QPointF, QSize
import fitz
from opengenai.langchain_ollama import CustomChatOllama


class PDFViewer(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc = None
        self.zoom = 1.0
        self.highlights = {}
        self.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.setWidget(self.content_widget)
        self.layout = QVBoxLayout(self.content_widget)

    def load_pdf(self, path):
        self.doc = fitz.open(path)
        self.render_pages()

    def render_pages(self):
        if not self.doc:
            return

        # Clear existing layout
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

        for page_num in range(len(self.doc)):
            page_widget = PDFPageWidget(self.doc[page_num], self.zoom, page_num)
            self.layout.addWidget(page_widget)

        self.content_widget.setLayout(self.layout)

    def zoom_in(self):
        self.zoom *= 1.2
        self.render_pages()

    def zoom_out(self):
        self.zoom /= 1.2
        self.render_pages()

    def get_highlighted_text(self):
        text = ""
        for page_num in range(self.layout.count()):
            page_widget = self.layout.itemAt(page_num).widget()
            text += page_widget.get_highlighted_text()
        return text

    def clear_highlights(self):
        for page_num in range(self.layout.count()):
            page_widget = self.layout.itemAt(page_num).widget()
            page_widget.clear_highlights()


class PDFPageWidget(QWidget):
    def __init__(self, page, zoom, page_num):
        super().__init__()
        self.page = page
        self.zoom = zoom
        self.page_num = page_num
        self.highlights = []
        self.selection_start = None
        self.selection_end = None
        self.setFixedSize(self.sizeHint())

    def paintEvent(self, event):
        painter = QPainter(self)

        # Render the page
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = self.page.get_pixmap(matrix=matrix)

        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        painter.drawImage(0, 0, img)

        # Draw highlights
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 0, 100))
        for rect in self.highlights:
            painter.drawRect(rect)

        # Draw current selection
        if self.selection_start and self.selection_end:
            painter.setPen(QPen(QColor(0, 0, 255), 1, Qt.DashLine))
            painter.setBrush(QColor(0, 0, 255, 50))
            rect = QRectF(self.selection_start, self.selection_end)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        self.selection_start = event.pos()
        self.selection_end = None
        self.update()

    def mouseMoveEvent(self, event):
        self.selection_end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.selection_start and self.selection_end:
            rect = QRectF(self.selection_start, self.selection_end).normalized()
            self.highlights.append(rect)
            self.selection_start = None
            self.selection_end = None
            self.update()

    def sizeHint(self):
        return QSize(int(self.page.rect.width * self.zoom), int(self.page.rect.height * self.zoom))

    def get_highlighted_text(self):
        text = ""
        for rect in self.highlights:
            text += self.page.get_text("text", clip=(rect.left() / self.zoom,
                                                     rect.top() / self.zoom,
                                                     rect.right() / self.zoom,
                                                     rect.bottom() / self.zoom))
        return text

    def clear_highlights(self):
        self.highlights.clear()
        self.update()


class ChatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # System prompt area
        prompt_layout = QHBoxLayout()
        self.system_prompt = QPlainTextEdit()
        self.system_prompt.setPlaceholderText("Enter your system prompt here...")
        self.system_prompt.setMaximumHeight(50)
        prompt_layout.addWidget(self.system_prompt, 2)

        self.prompt_combo = QComboBox()
        self.prompt_combo.addItems(['Custom', 'Explain', 'Summarize', 'Analyze'])
        self.prompt_combo.currentTextChanged.connect(self.update_system_prompt)
        prompt_layout.addWidget(self.prompt_combo, 1)

        layout.addLayout(prompt_layout)

        # Chat display area
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)

        self.regenerate_btn = QPushButton('Regenerate', self)
        self.explain_more_btn = QPushButton('Explain More', self)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.regenerate_btn)
        button_layout.addWidget(self.explain_more_btn)
        layout.addLayout(button_layout)

    def update_system_prompt(self, selected_option):
        if selected_option == 'Custom':
            self.system_prompt.clear()
            self.system_prompt.setPlaceholderText("Enter your system prompt here...")
        elif selected_option == 'Explain':
            self.system_prompt.setPlainText("Explain the following text in simple terms:")
        elif selected_option == 'Summarize':
            self.system_prompt.setPlainText("Summarize the following text concisely:")
        elif selected_option == 'Analyze':
            self.system_prompt.setPlainText("Analyze the key points in the following text:")

    def append_message(self, question, answer):
        self.chat_area.clear()
        md_content = f"## Question:\n\n{question}\n\n## Answer:\n\n{answer}"
        html_content = markdown.markdown(md_content)
        self.chat_area.setHtml(html_content)

    def enable_action_buttons(self, enable):
        self.regenerate_btn.setEnabled(enable)
        self.explain_more_btn.setEnabled(enable)


class InteractivePdfReader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.llm = CustomChatOllama(
            model="qwen2:7b-instruct",
            base_url="http://192.168.162.49:8888"
        )
        self.last_prompt = ""
        self.last_response = ""

    def initUI(self):
        self.setWindowTitle('Advanced Interactive PDF Reader')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # PDF Viewer (60% of width)
        pdf_container = QWidget()
        pdf_layout = QVBoxLayout(pdf_container)
        self.pdf_viewer = PDFViewer(self)
        pdf_layout.addWidget(self.pdf_viewer)

        control_layout = QHBoxLayout()
        pdf_layout.addLayout(control_layout)

        open_btn = QPushButton('Open PDF', self)
        open_btn.clicked.connect(self.open_pdf)
        control_layout.addWidget(open_btn)

        zoom_in_btn = QPushButton('Zoom In', self)
        zoom_in_btn.clicked.connect(self.pdf_viewer.zoom_in)
        control_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton('Zoom Out', self)
        zoom_out_btn.clicked.connect(self.pdf_viewer.zoom_out)
        control_layout.addWidget(zoom_out_btn)

        # Chat Area (40% of width)
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)

        self.chat_widget = ChatWidget()
        chat_layout.addWidget(self.chat_widget)

        process_btn = QPushButton('Process Selected Text', self)
        process_btn.clicked.connect(self.process_selected_text)
        chat_layout.addWidget(process_btn)

        # Add PDF Viewer and Chat Area to main layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(pdf_container)
        main_splitter.addWidget(chat_container)
        main_splitter.setSizes([720, 480])  # Set initial sizes (60% - 40%)
        main_layout.addWidget(main_splitter)

        self.chat_widget.regenerate_btn.clicked.connect(self.regenerate_response)
        self.chat_widget.explain_more_btn.clicked.connect(self.explain_more)
        self.chat_widget.enable_action_buttons(False)

    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.pdf_viewer.load_pdf(file_path)

    def process_selected_text(self):
        selected_text = self.pdf_viewer.get_highlighted_text()
        if not selected_text:
            self.chat_widget.append_message("Please highlight some text first.", "No text selected.")
            self.chat_widget.enable_action_buttons(False)
            return

        system_prompt = self.chat_widget.system_prompt.toPlainText()
        if not system_prompt:
            system_prompt = "Analyze the following text:"

        system_prompt += " and return in nice markdown style"
        self.last_prompt = f"{system_prompt}\n\n{selected_text}"
        self.generate_response()

    def generate_response(self):
        response = self.llm.invoke(self.last_prompt).content
        self.last_response = response
        self.chat_widget.append_message(f"{self.last_prompt[:100]}...", response)
        self.pdf_viewer.clear_highlights()
        self.chat_widget.enable_action_buttons(True)

    def regenerate_response(self):
        self.generate_response()

    def explain_more(self):
        explain_prompt = f"Based on the following context and response, please provide more detailed explanations and examples:\n\nContext: {self.last_prompt}\n\nPrevious response: {self.last_response}\n\nPlease explain in more detail and provide examples:"
        self.last_prompt = explain_prompt
        self.generate_response()


    # def process_selected_text(self):
    #     selected_text = self.pdf_viewer.get_highlighted_text()
    #     if not selected_text:
    #         self.chat_widget.append_message("Please highlight some text first.", "No text selected.")
    #         return
    #
    #     system_prompt = self.chat_widget.system_prompt.toPlainText()
    #     if not system_prompt:
    #         system_prompt = "Analyze the following text:"
    #
    #     system_prompt+="and return in nice markdown style"
    #     prompt = f"{system_prompt}\n\n{selected_text}"
    #     response = self.llm.invoke(prompt).content
    #
    #     self.chat_widget.append_message(f"{system_prompt}\n\n{selected_text[:100]}...", response)
    #     self.pdf_viewer.clear_highlights()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InteractivePdfReader()
    ex.show()
    sys.exit(app.exec_())
