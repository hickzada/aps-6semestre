# 👁️ VigIA - Sistema Inteligente de Iluminação por Visão Computacional e PDI

> **Atividades Práticas Supervisionadas (APS) – 5º/6º Semestre de Ciência da Computação (UNIP)**  
> **Tema Oficial:** *Aplicação de Técnicas de Processamento Digital de Imagens e Visão Computacional para Resolver Problemas de Sustentabilidade Ambiental.*

---

## 📌 1. Visão Geral do Projeto

O **VigIA** é uma aplicação computacional voltada à **eficiência energética predial e mitigação de pegada de carbono**, desenvolvida especificamente para resolver o problema do desperdício de energia elétrica em salas de aula, laboratórios e escritórios que permanecem iluminados sem ocupação real.

### Contexto e Relevância Ambiental
- No Brasil, edificações públicas e comerciais respondem por aproximadamente **47% de toda a energia elétrica consumida** (EPE / Procel).
- Ambientes corporativos e acadêmicos frequentemente mantêm sistemas de iluminação ligados por dezenas de horas ociosas (noites, madrugadas e finais de semana).
- Esse desperdício sobrecarrega a matriz elétrica e demanda o acionamento frequente de usinas termelétricas fósseis com alta emissão de dióxido de carbono ($CO_2$).

### A Solução VigIA
O sistema opera em uma arquitetura híbrida de duas camadas complementares:
1. **Processamento Digital de Imagens (PDI Artesanal)**: Algoritmos de visão de baixo nível desenvolvidos manualmente em Python para filtragem espacial, realce de contornos e segmentação por subtração de plano de fundo estático.
2. **Rede Neural Convolucional (Keras / TensorFlow)**: Uma mini-CNN treinada com dados locais para classificar e confirmar a presença humana nos quadros, eliminando falsos positivos decorrentes de sombras ou variações de iluminação externa.
3. **Mecanismo Sentinela (Sensor PIR)**: Estratégia de repouso onde a câmera permanece em stand-by quando a sala está desocupada, sendo reativada sob demanda por detecção de calor/movimento.

---

## ⚖️ Conformidade com as Exigências do Manual da UNIP

O projeto atende rigorosamente a todos os critérios estabelecidos nas Seções I e II do Manual de APS:

| Requisito do Manual | Implementação no VigIA | Localização no Código |
| :--- | :--- | :--- |
| **Item 3.a (Filtros de PDI desenvolvidos pelo grupo em Python)** | **Zero uso de filtros prontos de bibliotecas** (`cv2.GaussianBlur`, `cv2.Sobel`, `cv2.threshold`, etc.). Todos os operadores matemáticos foram implementados do zero: conversão ITU-R BT.601, convolução 2D discreta com padding por extensão, filtro Gaussiano $3 \times 3$, operadores diferenciais de Sobel ($G_x$ e $G_y$), subtração de fundo e abertura morfológica (erosão e dilatação). | [`prototipo_vigia.py`](prototipo_vigia.py) |
| **Item 3.b (Modelagem e Algoritmo de Inteligência)** | Implementação de uma **Rede Neural Convolucional em ~15 linhas essenciais em Keras/TensorFlow**, utilizando dataset balanceado com as classes `vazio` e `ocupado` e exportação do artefato `modelo_vigia.keras`. | [`treinar_modelo.py`](treinar_modelo.py) |
| **Item 4 (Testes e Resultados com Imagens Reais)** | Execução do sistema sobre imagens reais de escritório e geração das evidências visuais das 6 etapas intermediárias de processamento. | Pasta [`etapas_processamento/`](etapas_processamento/) |
| **Métricas de Sustentabilidade** | Modelo analítico que calcula energia total economizada ($kWh$), economia financeira estimada ($R\$$) e emissões de carbono evitadas ($kg \text{ de } CO_2$). | Módulo `gerar_relatorio_sustentabilidade` |

---

## 📂 Estrutura do Repositório

```text
aps-6semestre/
│
├── README.md                   # Documento de apresentação e guia de execução
├── DOCUMENTACAO_APS_VIGIA.md   # Documentação técnica e fundamentação teórica
├── prototipo_vigia.py          # Núcleo do sistema: PDI manual + inferência + sustentabilidade
├── treinar_modelo.py           # Treinamento da mini-CNN Keras (~15 linhas essenciais)
├── test_vigia.py               # Suíte formal de testes automatizados (23 testes, 100% pass)
├── modelo_vigia.keras          # Modelo convolucional serializado
│
├── dataset/                    # Dataset de treinamento supervisionado
│   ├── vazio/                  # Amostras de ambientes sem ocupação
│   └── ocupado/                # Amostras de ambientes com presença humana
│
├── imagens_reais/              # Fotografias reais de calibração e teste
│   ├── escritorio_vazio.jpg    # Referência de fundo estático
│   └── escritorio_com_pessoas.jpg # Cena com ocupação humana ativa
│
├── etapas_processamento/       # Evidências visuais geradas para a monografia acadêmica
│   ├── 1_tons_de_cinza.png     # Conversão fotométrica BT.601
│   ├── 2_suavizacao_gaussiana.png # Atenuação de ruído de alta frequência
│   ├── 3_realce_sobel.png      # Gradiente e realce de arestas/silhuetas
│   ├── 4_subtracao_fundo.png   # Diferença absoluta em relação ao fundo
│   ├── 5_binarizacao_limiar.png# Binarização estática por corte de limiar
│   └── 6_morfologia_final.png  # Abertura morfológica (erosão + dilatação)
│
└── documentos/                 # Diretório de apoio acadêmico
    ├── ManualAPS.pdf           # Manual oficial de orientações da UNIP
    ├── VigIA.pdf               # Apresentação do projeto em PDF
    └── VigIA_Pitch.pptx        # Apresentação de slides editável para o seminário
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Recomenda-se ambiente Python 3.9 ou superior com as seguintes dependências:
```bash
pip install pillow numpy tensorflow
```

### 2. Executar o Protótipo VigIA (Demonstração Completa)
Para simular os 4 ciclos operacionais (entrada de ocupantes, desocupação da sala, modo sentinela e religamento imediato):
```bash
python3 prototipo_vigia.py
```
O console exibirá o log detalhado do pipeline de PDI, a decisão probabilística da rede neural convolucional e o relatório consolidado de sustentabilidade energética.

### 3. Re-treinar a Mini-CNN do Zero (Opcional)
Caso queira demonstrar o processo de treinamento supervisionado para a banca:
```bash
python3 treinar_modelo.py
```
O script carregará as amostras de `dataset/`, treinará o modelo convolucional por 10 épocas e atualizará o arquivo `modelo_vigia.keras`.

### 4. Executar os Testes Automatizados de Qualidade
Para certificar a integridade matemática dos filtros e a correta tomada de decisão:
```bash
python3 test_vigia.py
```
> Resultado: **23 testes unitários e de integração executados com 100% de aprovação (código de saída 0)**.

---

## 🔬 Fundamentação do Pipeline de PDI

1. **Aquisição:** Carregamento de quadros RGB e normalização dimensional bilinear ($320 \times 240$).
2. **Conversão Fotométrica:** Aplicação da norma **ITU-R BT.601** ($Y = 0.299R + 0.587G + 0.114B$).
3. **Filtro Passa-Baixas Gaussiano:** Convolução espacial com máscara simétrica $3 \times 3$ ($\frac{1}{16} [1, 2, 1; 2, 4, 2; 1, 2, 1]$) para atenuação de ruído estocástico do sensor.
4. **Filtro Passa-Altas de Sobel:** Cálculo diferencial das componentes $G_x$ e $G_y$, gerando a magnitude do gradiente $\nabla I = \sqrt{G_x^2 + G_y^2}$ para extração de silhuetas.
5. **Segmentação por Subtração:** Operação diferencial absoluta entre o quadro suavizado e a imagem de calibração estática ($|\Delta I(x, y)|$).
6. **Limiarização Global (Thresholding):** Mapeamento bivalente com ponto de corte rígido ($\tau = 28$), isolando regiões candidatas.
7. **Morfologia Matemática (Abertura):** Convolução com elemento estruturante plano $3 \times 3$ para **Erosão** (remoção de ruídos residuais isolados) seguida de **Dilatação** (reconstituição da área do objeto).
8. **Classificação Neural:** Avaliação por tensor tridimensional na mini-CNN para validação de forma e emissão do sinal de controle das luminárias.

---

## 🎯 Roteiro Sugerido para Apresentação Oral (10 a 15 Minutos)

- **Minutos 0 a 3:** Apresentação do problema de sustentabilidade (desperdício elétrico em prédios, queima de combustíveis fósseis e custos operacionais).
- **Minutos 3 a 7:** Destaque do diferencial acadêmico: **filtros matemáticos desenvolvidos artesanalmente pelo grupo**, exibindo as 6 imagens de `etapas_processamento/`.
- **Minutos 7 a 10:** Apresentação da arquitetura da mini-CNN em Keras (~15 linhas) e demonstração da execução ao vivo de `prototipo_vigia.py`.
- **Minutos 10 a 12:** Análise dos resultados do Relatório de Sustentabilidade ($kWh$ poupados e redução de emissões de $CO_2$).
- **Minutos 12 a 15:** Sessão de perguntas e respostas com a banca avaliadora.
