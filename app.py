import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import subprocess
import os
import time
from PIL import Image
import threading

# Configuração da página
st.set_page_config(
    page_title="Reconhecimento de Imagem - Massaki",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da aplicação
st.title("🎯 Sistema de Reconhecimento de Imagem")
st.markdown("### Detecção em Tempo Real com Teachable Machine")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Função para carregar o modelo
@st.cache_resource
def load_keras_model():
    try:
        # Verifica se o arquivo existe antes de tentar carregar
        if not os.path.exists("keras_Model.h5"):
            st.error("❌ Arquivo 'keras_Model.h5' não encontrado na raiz do repositório.")
            st.error("📁 Certifique-se de que o arquivo está na mesma pasta que app.py no seu repositório Git.")
            return None
        
        model = load_model("keras_Model.h5", compile=False)
        st.success("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        return None

# Função para carregar as labels
@st.cache_data
def load_labels():
    try:
        # Verifica se o arquivo existe antes de tentar carregar
        if not os.path.exists("labels.txt"):
            st.error("❌ Arquivo 'labels.txt' não encontrado na raiz do repositório.")
            st.error("📁 Certifique-se de que o arquivo está na mesma pasta que app.py no seu repositório Git.")
            return []
        
        with open("labels.txt", "r") as f:
            class_names = f.readlines()
        st.success("✅ Labels carregadas com sucesso!")
        return class_names
    except Exception as e:
        st.error(f"❌ Erro ao carregar as labels: {e}")
        return []

# Função para executar msk02.py (opcional - só funciona se o arquivo existir)
def execute_msk02():
    try:
        if not os.path.exists("msk02.py"):
            st.warning("⚠️ Arquivo 'msk02.py' não encontrado. Funcionalidade de áudio/ação desabilitada.")
            return
        
        subprocess.run(["python", "msk02.py"], check=True)
        st.success("🎵 Arquivo msk02.py executado com sucesso!")
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Erro ao executar msk02.py: {e}")
    except FileNotFoundError:
        st.error("❌ Arquivo msk02.py não encontrado!")

# Função para processar a imagem
def process_image(image, model, class_names):
    # Redimensiona a imagem para 224x224 pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Converte para array numpy e normaliza
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_normalized = (image_array / 127.5) - 1
    
    # Faz a predição
    prediction = model.predict(image_normalized, verbose=0)
    index = np.argmax(prediction)
    raw_class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Extrai o nome da classe limpo
    clean_class_name = raw_class_name[2:].strip()
    
    return clean_class_name, confidence_score, image_resized

# Interface principal
def main():
    # Carrega o modelo e as labels
    model = load_keras_model()
    class_names = load_labels()
    
    # Verifica se os arquivos essenciais foram carregados
    if model is None or not class_names:
        st.error("🚫 **Sistema não pode iniciar**: Arquivos essenciais não encontrados.")
        
        with st.expander("📋 **Instruções para Correção**", expanded=True):
            st.markdown("""
            **Para que o aplicativo funcione, você precisa:**
            
            1. **Adicionar os arquivos ao seu repositório Git:**
               - `keras_Model.h5` (modelo treinado do Teachable Machine)
               - `labels.txt` (arquivo de classes do Teachable Machine)
               - `msk02.py` (opcional - para funcionalidade de áudio/ação)
            
            2. **Estrutura de arquivos esperada:**
               ```
               seu-repositorio/
               ├── app.py
               ├── requirements.txt
               ├── keras_Model.h5     ← Adicione este arquivo
               ├── labels.txt         ← Adicione este arquivo
               ├── msk02.py          ← Opcional
               └── .streamlit/
                   └── packages.txt   ← Se necessário
               ```
            
            3. **Faça commit e push dos arquivos:**
               ```bash
               git add keras_Model.h5 labels.txt msk02.py
               git commit -m "Adicionar arquivos do modelo"
               git push
               ```
            
            4. **Faça um novo deploy no Streamlit Cloud**
            """)
        
        return
    
    # Configurações na sidebar
    confidence_threshold = st.sidebar.slider(
        "🎯 Limite de Confiança (%)", 
        min_value=50, 
        max_value=100, 
        value=99, 
        step=1,
        help="Confiança mínima para executar ações automáticas"
    ) / 100.0
    
    auto_execute = st.sidebar.checkbox(
        "🎵 Executar msk02.py automaticamente", 
        value=True,
        help="Executa ação quando 'Massaki' for detectado com alta confiança"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Status do Sistema:**")
    
    # Colunas para layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Captura da Webcam")
        
        # Placeholder para a imagem da webcam
        image_placeholder = st.empty()
        
        # Botões de controle
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            start_button = st.button("▶️ Iniciar Captura", type="primary", use_container_width=True)
        
        with col_btn2:
            stop_button = st.button("⏹️ Parar Captura", use_container_width=True)
        
        with col_btn3:
            manual_execute = st.button("🎵 Executar Ação", use_container_width=True)
    
    with col2:
        st.subheader("📊 Resultados")
        
        # Placeholders para os resultados
        class_placeholder = st.empty()
        confidence_placeholder = st.empty()
        status_placeholder = st.empty()
    
    # Controle de estado da captura
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    
    if 'last_execution' not in st.session_state:
        st.session_state.last_execution = 0
    
    # Lógica dos botões
    if start_button:
        st.session_state.capturing = True
    
    if stop_button:
        st.session_state.capturing = False
    
    if manual_execute:
        execute_msk02()
    
    # Loop de captura
    if st.session_state.capturing:
        try:
            # Inicializa a câmera
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                st.error("📷 Não foi possível acessar a câmera. Verifique se ela está conectada e não está sendo usada por outro aplicativo.")
                st.session_state.capturing = False
                return
            
            while st.session_state.capturing:
                ret, frame = camera.read()
                
                if not ret:
                    st.error("❌ Erro ao capturar frame da câmera.")
                    break
                
                # Processa a imagem
                class_name, confidence, processed_image = process_image(frame, model, class_names)
                
                # Converte BGR para RGB para exibição no Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Exibe a imagem
                with image_placeholder.container():
                    st.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Atualiza os resultados
                with class_placeholder.container():
                    st.metric("🏷️ Classe Detectada", class_name)
                
                with confidence_placeholder.container():
                    confidence_percent = int(confidence * 100)
                    st.metric("📈 Confiança", f"{confidence_percent}%")
                    
                    # Barra de progresso para a confiança
                    st.progress(confidence)
                
                # Verifica se deve executar msk02.py
                current_time = time.time()
                if (class_name == "Massaki" and 
                    confidence >= confidence_threshold and 
                    auto_execute and 
                    current_time - st.session_state.last_execution > 5):  # Evita execuções muito frequentes
                    
                    with status_placeholder.container():
                        st.success(f"🎯 Classe 'Massaki' reconhecida com {confidence_percent}% de confiança!")
                        st.info("🎵 Executando ação...")
                    
                    # Executa em thread separada para não bloquear a interface
                    threading.Thread(target=execute_msk02, daemon=True).start()
                    st.session_state.last_execution = current_time
                
                # Atualiza status na sidebar
                st.sidebar.success("🟢 Sistema Ativo")
                st.sidebar.metric("🏷️ Última Detecção", class_name)
                st.sidebar.metric("📈 Confiança Atual", f"{confidence_percent}%")
                
                # Pequeno delay para não sobrecarregar
                time.sleep(0.1)
            
            # Libera a câmera
            camera.release()
            
        except Exception as e:
            st.error(f"❌ Erro durante a captura: {e}")
            st.session_state.capturing = False
    
    else:
        # Sistema parado
        st.sidebar.info("🔴 Sistema Parado")
        with image_placeholder.container():
            st.info("👆 Clique em '▶️ Iniciar Captura' para começar o reconhecimento em tempo real.")

# Informações adicionais
def show_info():
    st.markdown("---")
    st.subheader("ℹ️ Informações do Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📋 Requisitos:**
        - ✅ Streamlit Cloud
        - ✅ Webcam conectada
        - 📁 `keras_Model.h5`
        - 📁 `labels.txt`
        - 📁 `msk02.py` (opcional)
        """)
    
    with col2:
        st.markdown("""
        **🎯 Funcionalidades:**
        - 📹 Reconhecimento em tempo real
        - ⚙️ Ajuste de confiança
        - 🎵 Execução automática de ações
        - 🌐 Interface web interativa
        """)
    
    with col3:
        st.markdown("""
        **🔧 Controles:**
        - ▶️ Iniciar/Parar captura
        - 🎯 Ajustar limite de confiança
        - 🎵 Execução manual de ações
        - 📊 Visualização de resultados
        """)

    # Informações técnicas
    with st.expander("🔧 **Informações Técnicas**"):
        st.markdown("""
        **Dependências:**
        - `streamlit` - Interface web
        - `opencv-python-headless` - Processamento de imagem
        - `tensorflow/keras` - Modelo de IA
        - `numpy` - Computação numérica
        - `Pillow` - Manipulação de imagem
        
        **Compatibilidade:**
        - ✅ Streamlit Cloud
        - ✅ Modelos do Teachable Machine
        - ✅ Webcam via navegador
        - ✅ Execução de scripts Python
        """)

if __name__ == "__main__":
    main()
    show_info()
