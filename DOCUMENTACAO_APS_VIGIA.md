# VigIA - Sistema Inteligente de Ocupação e Eficiência Energética
## Documentação Técnica e Alinhamento com o Manual da APS (UNIP CC 5º/6º Semestre)

---

### 1. Resumo do Projeto e Alinhamento com a APS
- **Tema Oficial da APS**: *"Aplicação de Técnicas de Processamento Digital de Imagens e Visão Computacional para Resolver Problemas de Sustentabilidade Ambiental"*.
- **Problema Abordado**: Desperdício massivo de energia elétrica em edificações públicas e comerciais (salas de aula, escritórios, corredores) que permanecem com iluminação acesa 24 horas por dia sem ocupação real.
- **Impacto Ambiental**: Cerca de 47% do consumo elétrico no Brasil ocorre em edificações. O consumo ocioso demanda acionamento desnecessário de usinas termelétricas fósseis, sobrecarga hídrica e emissão de toneladas de dióxido de carbono ($CO_2$).
- **Solução VigIA**: Sistema de visão computacional em duas etapas que reaproveita a infraestrutura de câmeras de segurança existentes em conjunto com um sensor de infravermelho passivo (PIR) de baixo custo. O processamento digital de imagem confirma a presença humana real antes de ligar/manter a iluminação, eliminando falsos positivos e desligando as lâmpadas quando a sala está desocupada.

---

### 2. Conformidade Rigorosa com as Restrições do Manual da APS
O item **3.a do Manual da APS** estabelece expressamente:
> *"Os filtros devem ser desenvolvidos pelo grupo preferencialmente em linguagem Python e aplicados em imagens reais relacionadas ao problema escolhido."*

Para atender **estritamente** a essa exigência do orientador/banca da UNIP:
1. **Zero Bibliotecas Caixa-Preta de PDI**: Não foi utilizada nenhuma função pronta de filtros de bibliotecas como OpenCV (`cv2.GaussianBlur`, `cv2.Sobel`, `cv2.threshold`, `cv2.filter2D`, etc.) ou scikit-image.
2. **Implementação Algorítmica Manual**:
   - **Conversão para Escala de Cinza**: Implementação manual da fórmula de luminância $Y = 0.299R + 0.587G + 0.114B$ (norma ITU-R BT.601).
   - **Algoritmo de Convolução 2D**: Loop espacial completo com tratamento de borda (padding por extensão) e aplicação de máscaras matemáticas.
   - **Filtro Passa-Baixas Gaussiano 3x3**: Matriz de suavização espacial desenvolvida manualmente para redução de ruído térmico do sensor.
   - **Filtro Passa-Altas de Sobel**: Matrizes de gradiente horizontal ($G_x$) e vertical ($G_y$) calculadas via convolução manual e combinadas pela magnitude euclidiana $\sqrt{G_x^2 + G_y^2}$ para realce de bordas/silhuetas.
   - **Segmentação por Diferença de Fundo**: Subtração absoluta pixel a pixel entre o quadro atual e o quadro de calibração da sala vazia.
   - **Binarização (Thresholding)**: Limiarização manual com corte rígido $0$ ou $255$.
   - **Morfologia Matemática (Abertura)**: Algoritmos de erosão e dilatação 3x3 implementados artesanalmente para remover ruídos do tipo sal-e-pimenta.
3. **Uso Mínimo da Biblioteca Pillow (`PIL`)**: Utilizada **exclusivamente** como interface I/O para leitura (`Image.open`) e gravação (`Image.save`) de pixels no sistema operacional.

---

### 3. Estrutura da Pasta do Projeto (`vigia/`)

Tudo foi centralizado e organizado dentro da pasta `vigia/`:

```
vigia/
│
├── prototipo_vigia.py          # Código-fonte principal com PDI manual e imagens reais
├── DOCUMENTACAO_APS_VIGIA.md   # Documentação de conformidade com o Manual da APS
│
├── imagens_reais/              # Banco de imagens reais de escritório e pessoas
│   ├── escritorio_vazio.jpg    # Escritório vazio à noite (Fundo de calibração)
│   └── escritorio_com_pessoas.jpg # Pessoas trabalhando no escritório à noite
│
├── etapas_processamento/       # As 6 evidências visuais geradas pelo PDI manual
│   ├── 1_tons_de_cinza.png     # Conversão de luminância BT.601
│   ├── 2_suavizacao_gaussiana.png # Filtro passa-baixas Gaussiano 3x3 manual
│   ├── 3_realce_sobel.png      # Filtro gradiente Sobel 3x3 manual
│   ├── 4_subtracao_fundo.png   # Subtração de fundo pixel a pixel
│   ├── 5_binarizacao_limiar.png# Binarização rígida por limiar
│   └── 6_morfologia_final.png  # Abertura morfológica manual (erosão + dilatação)
│
└── documentos/                 # Manuais e slides originais
    ├── ManualAPS.pdf           # Manual oficial de APS da UNIP
    ├── VigIA.pdf               # Apresentação do projeto VigIA
    └── VigIA_Pitch.pptx        # Pitch de slides do VigIA
```

---

### 4. Como Executar o Protótipo

No terminal, dentro da pasta do projeto, execute:
```bash
python vigia/prototipo_vigia.py
```
ou, se você já estiver dentro da pasta `vigia`:
```bash
python prototipo_vigia.py
```

O programa executará automaticamente 4 ciclos de teste simulando a rotina de um dia de aula com imagens reais:
1. **Ciclo 1 (Ocupante presente na sala)**: VigIA processa a imagem real, detecta presença humana (2,75% de área ativa) com 87% de confiança e **mantém a iluminação acesa**. Todas as 6 etapas intermediárias de PDI são salvas em `vigia/etapas_processamento/`.
2. **Ciclo 2 (Ocupante sai da sala)**: VigIA processa o quadro, detecta sala vazia (0.0% de pixels ativos) e **apaga as luzes automaticamente**.
3. **Ciclo 3 (Madrugada / Fim de semana)**: Modo sentinela ativo. Câmera entra em repouso e lâmpadas permanecem apagadas.
4. **Ciclo 4 (Reentrada de pessoa com gatilho PIR)**: O sensor de infravermelho detecta calor/movimento, acorda o sistema de visão, confirma ocupação e **religa as luzes instantaneamente**.

Ao final, é impresso o **Relatório de Sustentabilidade** com a economia em kWh, valor em Reais (R$) e a redução de emissões de $CO_2$.

---

### 5. Como Testar com Suas Próprias Fotos de Celular

Basta colocar as fotos na pasta `vigia/imagens_reais/` e alterar a chamada no `prototipo_vigia.py`:
```python
sistema.calibrar_fundo("imagens_reais/sua_foto_vazia.jpg")
sistema.processar_ciclo("imagens_reais/sua_foto_com_gente.jpg", sinal_sensor_pir=True)
```
*(O sistema possui redimensionamento automático bilinear durante a aquisição, aceitando fotos de câmeras de alta resolução de 12MP/4K sem sobrecarregar a memória nem deixar o processamento lento).*
