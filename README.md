# VigIA - Sistema de Iluminação por PDI e Visão Computacional

Trabalho de Atividades Práticas Supervisionadas (APS) – 6º Semestre de Ciência da Computação (UNIP).  
Tema: Aplicação de Técnicas de Processamento Digital de Imagens e Visão Computacional para Resolver Problemas de Sustentabilidade Ambiental.

---

## Integrantes do Grupo

- Adrian
- Gabriel
- Henrique
- Isa
- Sarah

---

## Como Rodar o Projeto

O modelo neural (`modelo_vigia.keras`) já está treinado e versionado no repositório. Não é necessário treinar nada antes de rodar.

### 1. Instalação das Dependências

No terminal ou prompt de comando na pasta do projeto:

```bash
pip install pillow numpy tensorflow pygame
```

*Nota: Conforme a restrição do Manual da APS (item 3.a), o projeto **não utiliza OpenCV** (`cv2`). Toda a esteira de convolução, filtragem e morfologia foi desenvolvida artesanalmente em Python com NumPy.*

### 2. Executar a Interface com Webcam

Para abrir o console com o feed da câmera e a máscara morfológica em tempo real:

```bash
python3 interface_webcam.py
```

**Instruções de uso na câmera:**
- Ao abrir, afaste-se do enquadramento e pressione **`[ESPAÇO]`** para registrar o plano de fundo da sala vazia.
- **Detecção Humana:** Ao entrar na frente da câmera, o sistema identifica a presença corporal e a biometria facial/pele, desenhando a caixa verde (`HUMANO CONFIRMADO`) e acionando a carga de iluminação (400W).
- **Rejeição de Objetos:** Se um celular, garrafa, mochila ou cadeira for colocado na frente da lente, o sistema classifica o alvo como objeto inanimado, desenhando uma caixa amarela (`OBJETO DETECTADO: IGNORADO`), mantendo o relé em standby (0W).
- Pressione **`[ESC]`** ou **`[Q]`** para fechar.

### 3. Executar o Protótipo no Terminal (Headless)

Simula o ciclo operacional completo com imagens estáticas de escritório e imprime o balanço de sustentabilidade energética (kWh economizados, custo evitado em R$ e redução de CO2):

```bash
python3 prototipo_vigia.py
```

### 4. Executar os Testes Automatizados

Para certificar que todos os filtros matemáticos, binarização, morfologia e inferência neural estão funcionando:

```bash
python3 test_vigia.py
```

Total de 23 testes unitários e de integração (100% de aprovação).

### 5. Re-treinar a Rede Neural (Opcional)

Se alguém quiser re-treinar a mini-CNN do zero:

```bash
python3 treinar_modelo.py
```

O script lê as amostras de `dataset/`, executa o treinamento supervisionado e salva o novo `modelo_vigia.keras`. Depois é só commitar o arquivo gerado.

---

## Próximos Passos e Divisão de Tarefas da Equipe

Com o código e os testes finalizados, restam as seguintes etapas de conferência e entrega:

1. **Gabriel:**
   - Clonar o repositório em uma máquina limpa e rodar `pip install -r` (ou os comandos acima) para certificar que roda de primeira sem conflitos de biblioteca.
   - Executar `python3 test_vigia.py` e confirmar que os 23 testes passam.

2. **Isa:**
   - Fazer a revisão final do relatório escrito em PDF, verificando as normas ABNT exigidas pelo manual da UNIP: fonte Arial 12, espaçamento 1.5, margens 2,5 cm e numeração de páginas.
   - Conferir se os nomes e RAs de todos os cinco integrantes estão corretos na capa e folha de rosto.

3. **Adrian:**
   - Conferir se as imagens salvas na pasta `etapas_processamento/` (tons de cinza, Gaussiano, Sobel, subtração, limiar, morfologia e o `painel_comparativo_pdi.png`) estão inseridas no capítulo de Desenvolvimento do relatório como figuras numeradas.

4. **Sarah:**
   - Imprimir e preencher a Ficha de Atividades Práticas Supervisionadas padrão da UNIP (anexa no final do manual da APS).
   - Coletar as assinaturas dos integrantes para anexar junto ao trabalho.

5. **Henrique / Grupo:**
   - Realizar o upload da versão final do relatório em PDF no portal de entrega da UNIP ([http://trabalhosacademicos.unip.br/entrega/](http://trabalhosacademicos.unip.br/entrega/)) antes do encerramento do prazo.

---

## Estrutura do Código

```text
aps-6semestre/
├── interface_webcam.py         # Console gráfico 60 FPS com Pygame (feed ao vivo + máscara)
├── prototipo_vigia.py          # Pipeline manual de PDI (convolução 2D, Gauss, Sobel, morfologia)
├── treinar_modelo.py           # Script de treino supervisionado da mini-CNN Keras
├── test_vigia.py               # Suíte de testes com 23 asserções automatizadas
├── modelo_vigia.keras          # Rede neural convolucional treinada e compilada
├── dataset/                    # Base local de imagens para treino (vazio e ocupado)
├── imagens_reais/              # Fotos de calibração e teste de escritório
├── etapas_processamento/       # Imagens intermediárias geradas pelos filtros manuais
└── documentos/                 # Manual da APS da UNIP e materiais complementares
```

---

## Resumo Técnico dos Filtros (Conformidade UNIP Item 3.a)

- **Luminância (Tons de Cinza):** ITU-R BT.601 ($Y = 0.299R + 0.587G + 0.114B$).
- **Filtro Gaussiano 3x3:** Convolução espacial 2D com máscara simétrica de suavização para redução de ruído térmico.
- **Realce Sobel 3x3:** Operadores diferenciais $G_x$ e $G_y$ combinados por magnitude vetorial $\sqrt{G_x^2 + G_y^2}$.
- **Subtração de Fundo:** Diferença absoluta $|\Delta I(x, y)|$ pixel a pixel em relação ao quadro de calibração.
- **Binarização (Thresholding):** Corte rígido com limiar empírico ($\tau = 28$).
- **Morfologia Matemática:** Abertura estruturada (Erosão 3x3 seguida de Dilatação 3x3) para eliminação de ruídos espúrios sem perda da silhueta corporal.
- **Classificação Neural & Biometria:** Mini-CNN Keras operando sobre o recorte (Bounding Box) do alvo em conjunto com o modelo fotométrico de segmentação de pele (Kovac et al.), garantindo rejeição de celulares, móveis e objetos inanimados.
