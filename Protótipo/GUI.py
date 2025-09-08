import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QProgressBar,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                             QFileDialog, QGroupBox, QFormLayout, QSpinBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importar sua classe existente
from main import RedditDataProcessor  # Assumindo que seu arquivo se chama reddit_analyzer.py

class AnalysisThread(QThread):
    """Thread para executar a análise em segundo plano"""
    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(bool, dict, pd.DataFrame)
    error_signal = pyqtSignal(str)
    
    def __init__(self, processor, subreddit, sample_size):
        super().__init__()
        self.processor = processor
        self.subreddit = subreddit
        self.sample_size = sample_size
    
    def run(self):
        try:
            self.progress_signal.emit("Iniciando análise...", 10)
            
            # Processar dados
            success = self.processor.process_data(self.subreddit, self.sample_size)
            
            if success:
                self.progress_signal.emit("Gerando visualizações...", 70)
                self.processor.generate_visualizations("visualizations/")
                
                self.progress_signal.emit("Gerando relatório...", 90)
                report_file = f"report_{self.subreddit}.txt"
                self.processor.generate_report(report_file)
                
                self.progress_signal.emit("Análise concluída!", 100)
                self.finished_signal.emit(True, self.processor.stats, self.processor.processed_data)
            else:
                self.error_signal.emit("Falha no processamento dos dados")
                
        except Exception as e:
            self.error_signal.emit(f"Erro durante a análise: {str(e)}")

class MplCanvas(FigureCanvas):
    """Widget para exibir gráficos matplotlib"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

class RedditAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = RedditDataProcessor()
        self.current_data = pd.DataFrame()
        self.current_stats = {}
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('NEXPOL - Analisador de Polarização do Reddit')
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central e layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Splitter para dividir a interface
        splitter = QSplitter(Qt.Horizontal)
        
        # Painel esquerdo - Controles
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Grupo de entrada
        input_group = QGroupBox("Configuração da Análise")
        input_layout = QFormLayout()
        
        self.subreddit_input = QLineEdit("politics")
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(10, 500)
        self.sample_size_spin.setValue(50)
        
        input_layout.addRow("Subreddit:", self.subreddit_input)
        input_layout.addRow("Tamanho da Amostra:", self.sample_size_spin)
        input_group.setLayout(input_layout)
        
        # Botões
        self.analyze_btn = QPushButton("Iniciar Análise")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        self.export_btn = QPushButton("Exportar Dados")
        self.export_btn.clicked.connect(self.export_data)
        self.export_btn.setEnabled(False)
        
        left_layout.addWidget(input_group)
        left_layout.addWidget(self.analyze_btn)
        left_layout.addWidget(self.export_btn)
        left_layout.addStretch()
        
        # Painel direito - Resultados
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Abas para diferentes visualizações
        self.tabs = QTabWidget()
        
        # Aba de estatísticas
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        # Aba de dados
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        self.data_table = QTableWidget()
        data_layout.addWidget(self.data_table)
        
        # Aba de gráficos
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        self.graph_canvas = MplCanvas(self, width=8, height=6)
        graph_layout.addWidget(self.graph_canvas)
        
        self.tabs.addTab(stats_tab, "Estatísticas")
        self.tabs.addTab(data_tab, "Dados")
        self.tabs.addTab(graph_tab, "Gráficos")
        
        right_layout.addWidget(self.tabs)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage('Pronto para analisar')
        
    def start_analysis(self):
        subreddit = self.subreddit_input.text().strip()
        sample_size = self.sample_size_spin.value()
        
        if not subreddit:
            QMessageBox.warning(self, "Erro", "Por favor, digite um nome de subreddit")
            return
        
        # Desabilitar botão durante análise
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage(f'Analisando r/{subreddit}...')
        
        # Criar e iniciar thread de análise
        self.analysis_thread = AnalysisThread(self.processor, subreddit, sample_size)
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.finished_signal.connect(self.analysis_finished)
        self.analysis_thread.error_signal.connect(self.analysis_error)
        self.analysis_thread.start()
    
    def update_progress(self, message, value):
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def analysis_finished(self, success, stats, data):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        
        self.current_stats = stats
        self.current_data = data
        
        self.display_results()
        self.statusBar().showMessage('Análise concluída com sucesso!')
        
        QMessageBox.information(self, "Sucesso", 
                               f"Análise de r/{stats['subreddit']} concluída!\n"
                               f"Score de polarização: {stats['polarization_score']:.4f}")
    
    def analysis_error(self, error_message):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erro", error_message)
        self.statusBar().showMessage('Erro na análise')
    
    def display_results(self):
        # Exibir estatísticas
        stats_text = f"""
        RELATÓRIO DE ANÁLISE - r/{self.current_stats['subreddit']}
        {'='*50}
        
        📊 ESTATÍSTICAS GERAIS:
        • Total de posts: {self.current_stats['total_posts']}
        • Score médio: {self.current_stats['avg_score']:.2f}
        • Comentários médios: {self.current_stats['avg_comments']:.2f}
        
        🎯 ANÁLISE DE POLARIZAÇÃO:
        • Score de polarização: {self.current_stats['polarization_score']:.4f}
        • Nível: {self.current_stats['polarization_level']}
        
        😊 DISTRIBUIÇÃO DE SENTIMENTOS:
        • Positivos: {self.current_stats['positive_posts']} ({self.current_stats['positive_posts']/self.current_stats['total_posts']*100:.1f}%)
        • Neutros: {self.current_stats['neutral_posts']} ({self.current_stats['neutral_posts']/self.current_stats['total_posts']*100:.1f}%)
        • Negativos: {self.current_stats['negative_posts']} ({self.current_stats['negative_posts']/self.current_stats['total_posts']*100:.1f}%)
        • Sentimento médio: {self.current_stats['avg_sentiment']:.4f}
        """
        
        self.stats_text.setPlainText(stats_text)
        
        # Exibir dados na tabela
        self.display_data_table()
        
        # Exibir gráfico
        self.display_graph()
    
    def display_data_table(self):
        if self.current_data.empty:
            return
        
        # Configurar tabela
        self.data_table.setRowCount(len(self.current_data))
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(['Título', 'Autor', 'Score', 'Comentários', 'Sentimento', 'Data'])
        
        # Preencher tabela
        for row_idx, (_, row) in enumerate(self.current_data.iterrows()):
            self.data_table.setItem(row_idx, 0, QTableWidgetItem(str(row['title'])[:50] + '...'))
            self.data_table.setItem(row_idx, 1, QTableWidgetItem(str(row['author'])))
            self.data_table.setItem(row_idx, 2, QTableWidgetItem(str(row['score'])))
            self.data_table.setItem(row_idx, 3, QTableWidgetItem(str(row['num_comments'])))
            self.data_table.setItem(row_idx, 4, QTableWidgetItem(f"{row['sentiment_compound']:.4f}"))
            self.data_table.setItem(row_idx, 5, QTableWidgetItem(str(row['created_utc'].date())))
        
        # Ajustar tamanho das colunas
        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
    
    def display_graph(self):
        # Limpar gráfico anterior
        self.graph_canvas.axes.clear()
        
        if self.current_data.empty:
            return
        
        # Criar gráfico de distribuição de sentimentos
        sentiment_counts = [
            self.current_stats['positive_posts'],
            self.current_stats['neutral_posts'], 
            self.current_stats['negative_posts']
        ]
        labels = ['Positivo', 'Neutro', 'Negativo']
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        self.graph_canvas.axes.pie(sentiment_counts, labels=labels, colors=colors, 
                                  autopct='%1.1f%%', startangle=90)
        self.graph_canvas.axes.set_title(f'Distribuição de Sentimentos - r/{self.current_stats["subreddit"]}')
        
        self.graph_canvas.draw()
    
    def export_data(self):
        if self.current_data.empty:
            QMessageBox.warning(self, "Erro", "Nenhum dado para exportar")
            return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Dados", "reddit_analysis.csv", 
            "CSV Files (*.csv);;Excel Files (*.xlsx)", options=options
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.current_data.to_csv(file_path, index=False, encoding='utf-8')
                else:
                    self.current_data.to_excel(file_path, index=False)
                
                QMessageBox.information(self, "Sucesso", f"Dados exportados para: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao exportar dados: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Configurar estilo da aplicação
    app.setStyle('Fusion')
    
    window = RedditAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()