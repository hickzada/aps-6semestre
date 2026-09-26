# 👁️ VigIA - Sistema Inteligente de Iluminação por Visão Computacional
> **Projeto de Atividades Práticas Supervisionadas (APS) – 5º/6º Semestre de Ciência da Computação (UNIP)**  
> **Tema Oficial:** *Aplicação de Técnicas de Processamento Digital de Imagens e Visão Computacional para Resolver Problemas de Sustentabilidade Ambiental.*

---

## 📌 1. Sobre o Projeto

O **VigIA** é uma solução para combater o **desperdício de energia elétrica em prédios comerciais, faculdades e escolas**. 

### O Problema Real
* Segundo a EPE (Empresa de Pesquisa Energética), **47% de toda a energia elétrica consumida no Brasil é gasta dentro de edificações**.
* Andares inteiros de escritórios e salas de aula ficam com luzes 100% acesas 24h por dia, mesmo vazios, à noite e aos finais de semana.
* Esse desperdício força o acionamento de **usinas termelétricas fósseis**, gerando emissões massivas de carbono ($CO_2$) e custos financeiros desnecessários.

### A Solução do VigIA
Em vez de sensores de presença comuns (que apagam a luz na cara das pessoas se elas ficarem paradas) ou de instalar câmeras caras em cada lâmpada:
1. **Reaproveita câmeras de segurança (CFTV)** já existentes na sala.
2. Usa **Processamento Digital de Imagens (PDI)** para confirmar presença humana real (diferencia pessoas de sombras/insetos).
3. Usa um **sensor infravermelho passivo (PIR)** barato como "sentinela": quando a sala esvazia e a luz apaga, a câmera dorme. Se alguém entra, o sensor PIR acorda a câmera, valida a imagem e acende a luz na hora.

---

## ⚠️ REGRA CRÍTICA DO PROFESSOR / MANUAL DA APS

> 🚨 **MUITO IMPORTANTE PARA QUEM FOR MEXER NO CÓDIGO:**
> 
> O item **3.a do Manual da APS** diz expressamente:
> *"Os filtros devem ser desenvolvidos pelo grupo preferencialmente em linguagem Python"*.
> 
> Isso significa que **NÃO PODEMOS USAR filtros prontos de bibliotecas** como OpenCV (`cv2.GaussianBlur`, `cv2.Sobel`, `cv2.threshold`, etc.).
> 
> Se usarmos as funções prontas do OpenCV, o professor pode zerar essa parte. Por isso, **todos os algoritmos matemáticos foram implementados na mão no código**:
> - Convolução 2D espacial com tratamento de borda.
> - Luminância de tons de cinza ($Y = 0.299R + 0.587G + 0.114B$).
> - Filtro Gaussiano 3x3 de suavização.
> - Filtro de Sobel horizontal ($G_x$) e vertical ($G_y$).
> - Subtração de fundo pixel a pixel.
> - Limiarização (binarização).
> - Operações morfológicas de erosão e dilatação (abertura).
> 
> A biblioteca **Pillow (`PIL`)** é usada **única e exclusivamente** para carregar os pixels da foto (`Image.open`) e salvar as imagens no disco (`Image.save`).

---

## 📂 Estrutura das Pastas

Tudo o que precisamos está centralizado nesta pasta:

```text
vigia/
│
├── README.md                   # Este guia rápido para o grupo
├── prototipo_vigia.py          # Código Python principal do sistema
├── DOCUMENTACAO_APS_VIGIA.md   # Relatório técnico completo alinhado ao manual
│
├── imagens_reais/              # Fotos reais de teste
│   ├── escritorio_vazio.jpg    # Sala/escritório vazio (Fundo de calibração)
│   └── escritorio_com_pessoas.jpg # Sala/escritório com pessoas trabalhando
│
├── etapas_processamento/       # Imagens que o código gera para o nosso relatório escrito!
│   ├── 1_tons_de_cinza.png     # Foto convertida em escala de cinza
│   ├── 2_suavizacao_gaussiana.png # Foto após filtro Gaussiano (reduz ruído)
│   ├── 3_realce_sobel.png      # Foto com realce de contornos (Sobel)
│   ├── 4_subtracao_fundo.png   # Diferença entre a sala vazia e a sala ocupada
│   ├── 5_binarizacao_limiar.png# Binarização (pixels brancos = presença)
│   └── 6_morfologia_final.png  # Limpeza de ruídos por abertura morfológica
│
└── documentos/                 # Nossos materiais de entrega da UNIP
    ├── ManualAPS.pdf           # O manual oficial de regras e prazos da APS
    ├── VigIA.pdf               # Slides do projeto
    └── VigIA_Pitch.pptx        # Apresentação editável de slides
```

---

## 🚀 Como Rodar o Projeto no Seu Computador

### 1. Pré-requisitos
Apenas o **Python 3** instalado.

### 2. Instalar a biblioteca de imagens
Abra o terminal (Prompt de Comando ou PowerShell) e instale o Pillow:
```bash
pip install pillow
```

### 3. Executar o protótipo
Entre na pasta do projeto e rode:
```bash
python prototipo_vigia.py
```

O programa vai rodar uma simulação completa de 4 ciclos e exibir no terminal:
- A leitura das imagens reais.
- A contagem de pixels ativos e o percentual de ocupação da sala.
- A decisão tomada pela IA (Manter ligada / Apagar / Religar).
- O **Relatório de Sustentabilidade** (kWh economizados, reais poupados e kg de $CO_2$ evitados).

---

## 🖼️ Como Testar com Novas Fotos (Suas Fotos de Celular)

Qualquer membro do grupo pode testar com fotos tiradas da sua sala, quarto ou sala de aula da UNIP:

1. Tire uma foto da sala **vazia** (com as luzes acesas) e salve na pasta `imagens_reais/` com o nome `minha_sala_vazia.jpg`.
2. Tire uma foto da mesma sala **com você ou amigos dentro** e salve como `minha_sala_ocupada.jpg`.
3. Abra o arquivo `prototipo_vigia.py`, vá até o final (linhas 518-520) e altere:

```python
img_vazio = os.path.join(pasta_imagens_reais, "minha_sala_vazia.jpg")
img_ocupado = os.path.join(pasta_imagens_reais, "minha_sala_ocupada.jpg")
```

> 💡 **Dica técnica:** O código tem redimensionamento bilinear automático para `320x240`. Não precisa diminuir a resolução no celular; pode colocar fotos em 4K ou 12MP que o código trata sozinho sem travar!

---

## 🧠 Como Funciona o Pipeline de PDI (Resumo para Apresentação Oral)

Caso o professor pergunte no seminário como o código funciona, a resposta está nesta ordem:

1. **Aquisição:** Carrega a imagem RGB e ajusta para resolução padrão.
2. **Manipulação (Escala de Cinza):** Aplica a fórmula de luminância do olho humano ($0.299R + 0.587G + 0.114B$) para reduzir os 3 canais de cor para apenas 1 matriz de intensidades de 0 a 255.
3. **Filtragem Passa-Baixas (Gaussiano 3x3):** Aplica convolução manual com máscara gaussiana para atenuar o ruído eletrônico da câmera.
4. **Filtragem Passa-Altas (Sobel 3x3):** Aplica máscaras de gradiente horizontal ($G_x$) e vertical ($G_y$) para destacar as bordas e silhuetas de corpos e móveis.
5. **Segmentação (Subtração de Fundo):** Compara a matriz atual com a matriz de referência do ambiente vazio ($|Atual - Fundo|$).
6. **Limiarização:** Se a variação de luminosidade for maior que o limiar (ex: 28), o pixel vira $255$ (branco); senão vira $0$ (preto).
7. **Morfologia (Abertura):** Executa uma erosão seguida de dilatação para remover "pontinhos" soltos causados por poeira ou oscilação de luz.
8. **Decisão:** Se a área branca for superior ao percentual mínimo (ex: $0.8\%$), confirma a presença humana e aciona os relés de iluminação.

---

## 📋 Próximos Passos para o Grupo (Divisão de Tarefas)

Conforme o Manual da APS (páginas 3 a 5), o trabalho tem duas fases e avaliação dividida em:
- **Trabalho Escrito (35%)**
- **Programa Desenvolvido (35%)** *(Protótipo pronto aqui!)* Precisamos terminar
- **Apresentação em Sala (30%)**

### O que ainda precisamos fazer juntos:
- [x] Desenvolver o protótipo com filtros manuais em Python.
- [x] Testar com imagens reais e gerar imagens das 6 etapas de PDI.
- [ ] **Trabalho Escrito (ABNT):**
  - Formatação obrigatória: Fonte Arial 12, espaçamento 1,5, margens 2,5 cm, formato A4.
  - Preencher Capítulo 1 (Introdução e Contexto Ambiental).
  - Preencher Capítulo 2 (Referencial Teórico: Aquisição, Manipulação, Filtragem, Segmentação).
  - Preencher Capítulo 3 (Proposta dos Filtros e justificativa).
  - Preencher Capítulo 4 (Resultados obtidos com as fotos da pasta `etapas_processamento/`).
  - Preencher Apêndice com o código-fonte (`prototipo_vigia.py`).
- [X] **Apresentação Oral (10 a 15 minutos):**
  - Adaptar o pitch do arquivo `documentos/VigIA_Pitch.pptx`.
  - Distribuir a fala igualmente entre todos os membros (a participação individual vale 20% da nota da apresentação).
- [ ] **Ficha de APS:**
  - Preencher a folha de horas e atividades cronológicas (página 6 do manual).
