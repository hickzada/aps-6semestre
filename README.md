# 👁️ VigIA - Sistema Inteligente de Iluminação por Visão Computacional e PDI

> **Atividades Práticas Supervisionadas (APS) – 5º/6º Semestre de Ciência da Computação (UNIP)**  
> **Tema Oficial:** *Aplicação de Técnicas de Processamento Digital de Imagens e Visão Computacional para Resolver Problemas de Sustentabilidade Ambiental.*  
> **Status do Projeto:** ✅ 100% Concluído | 23/23 Testes Automatizados Aprovados | Zero Bibliotecas Caixa-Preta (Sem OpenCV)

---

## ⚡ Guia Rápido: Como Rodar no Computador de Qualquer Membro

O modelo de Inteligência Artificial **já está treinado e incluído no repositório** (`modelo_vigia.keras`). **Ninguém precisa treinar nada para rodar!** É só clonar, instalar os pacotes e executar.

### 1. Instalar as dependências necessárias
Abra o terminal (ou Prompt de Comando) na pasta do projeto e instale os pacotes básicos:
```bash
pip install pillow numpy tensorflow pygame
```
*(Lembrete importante da banca: **NÃO** usa OpenCV / `cv2`! Todos os filtros de imagem foram feitos artesanalmente em Python).*

### 2. Rodar a Interface ao Vivo com a Webcam (Demonstração Prática)
Para abrir o console visual em 60 FPS com a câmera do notebook ou webcam USB:
```bash
python3 interface_webcam.py
```
**Como usar a interface:**
1. Aproxime a câmera e garanta que o enquadramento esteja visível.
2. Afaste-se da câmera por 1 segundo e pressione **`[ESPAÇO]`** para calibrar o ambiente vazio.
3. **Teste do Humano:** Sente-se na cadeira ou entre no campo de visão. O sistema desenhará uma caixa **VERDE** com o rótulo `[HUMANO CONFIRMADO]` e acionará a carga (`CARGA: 400W [ACIONADA]`).
4. **Teste do Objeto / Celular:** Segure o smartphone, uma mochila ou uma garrafa na frente da câmera sem o seu rosto/corpo estar enquadrado. O sistema desenhará uma caixa **AMARELA** com o rótulo `[OBJETO DETECTADO: IGNORADO]`, indicando `BIOMETRIA PELE: 0.0%`, e manterá as luzes em `CARGA: 0W [STANDBY]`.
5. Pressione **`[ESC]`** ou **`[Q]`** para sair e ver o relatório final de $kWh$ poupados e $CO_2$ reduzido.

### 3. Rodar a Simulação Automática em Terminal (Sem Câmera)
Se quiser rodar os 4 ciclos prediais com as imagens reais de escritório:
```bash
python3 prototipo_vigia.py
```

### 4. Rodar os Testes de Validação (Garantia de Nota Máxima)
Para certificar que todos os filtros matemáticos, binarização, morfologia e rede neural estão funcionando:
```bash
python3 test_vigia.py
```
> Resultado esperado: **Ran 23 tests ... OK** (100% de aprovação).

---

## 🧠 Como Re-Treinar o Modelo e Subir Novo Commit (Se Alguém Quiser Treinar)

O arquivo `modelo_vigia.keras` já vem pronto e treinado. Porém, se alguém do grupo quiser treinar uma nova versão no seu computador e subir para o GitHub:

1. Execute o script de treinamento:
   ```bash
   python3 treinar_modelo.py
   ```
   *(O script lê o dataset local em `dataset/`, treina a mini-CNN Keras por 12 épocas em ~3 segundos e salva automaticamente o novo `modelo_vigia.keras`).*
2. Em seguida, envie o modelo treinado para o GitHub:
   ```bash
   git add modelo_vigia.keras dataset/
   git commit -m "chore: atualiza modelo neural treinado com novas amostras"
   git push origin main
   ```
3. Pronto! Assim que subir, qualquer outro membro do grupo que der `git pull` já receberá o modelo atualizado pronto para uso.

---

## 📋 TO-DO LIST PARA O GRUPO DA APS (O que Cada Membro Deve Fazer)

Use esta lista para organizar a divisão de tarefas do grupo até o dia da entrega e apresentação:

- [ ] **Membro 1 (Ambiente e Código):**
  - Clonar o repositório no computador que será levado para a faculdade.
  - Testar a execução de `python3 interface_webcam.py` na webcam do notebook para garantir que a iluminação da sala da faculdade funcione bem.
  - Rodar `python3 test_vigia.py` e tirar um print do terminal com o `OK` dos 23 testes para colocar nos slides.
- [ ] **Membro 2 (Slides e Apresentação):**
  - Abrir o arquivo [`documentos/VigIA_Pitch.pptx`](documentos/VigIA_Pitch.pptx) e inserir os nomes e RAs de todos os integrantes do grupo.
  - Adicionar o painel comparativo das 6 etapas de PDI que está salvo em [`etapas_processamento/painel_comparativo_pdi.png`](etapas_processamento/painel_comparativo_pdi.png).
  - Incluir os dados de sustentabilidade (economia de energia em $kWh$, economia financeira em $R\$$ e $kg \text{ de } CO_2$ evitados).
- [ ] **Membro 3 (Ficha da APS):**
  - Preencher a **Ficha Padrão de Atividades Práticas Supervisionadas** da UNIP (presente no final de [`documentos/ManualAPS.pdf`](documentos/ManualAPS.pdf)).
  - Coletar as assinaturas de todos os membros do grupo.
- [ ] **Membro 4 (Postagem no Portal UNIP):**
  - Acessar o sistema oficial de entrega da UNIP: [http://trabalhosacademicos.unip.br/entrega/](http://trabalhosacademicos.unip.br/entrega/).
  - Fazer o upload do trabalho em PDF antes da data limite do calendário escolar (a não postagem no prazo acarreta reprovação automática pelo sistema).

---

## 🎯 Roteiro da Apresentação em Sala de Aula (10 a 15 Minutos)

O Manual da APS estabelece que a apresentação tem peso de **30% da nota final**. Sugestão de divisão de falas para até 4 alunos:

| Tempo | Integrante | Conteúdo da Fala |
| :--- | :--- | :--- |
| **0 a 3 min** | Aluno 1 | **Introdução & Sustentabilidade:** O problema do desperdício de iluminação predial (47% do consumo no Brasil), acionamento de termelétricas fósseis e a proposta do VigIA em unir PDI artesanal + IA para desligar luzes ociosas. |
| **3 a 7 min** | Aluno 2 | **Processamento Digital de Imagens Manual:** Enfatizar à banca que **não usamos OpenCV**. Explicar as etapas desenvolvidas em Python puro: Conversão ITU-R BT.601, Convolução 2D, Gauss 3x3, Sobel, Subtração de fundo e Abertura Morfológica (mostrar a imagem [`painel_comparativo_pdi.png`](etapas_processamento/painel_comparativo_pdi.png)). |
| **7 a 10 min** | Aluno 3 | **Modelagem Inteligente & Discriminação:** Explicar a mini-CNN Keras (~15 linhas), como ela confirma a presença humana e o diferencial da **biometria cromática de pele (Kovac et al.)**, que impede que celulares, mochilas ou cadeiras liguem a luz indevidamente. |
| **10 a 12 min** | Aluno 4 / Grupo | **Demonstração Prática ao Vivo:** Abrir `python3 interface_webcam.py` no projetor. Calibrar o fundo, sentar para acender em verde (`HUMANO CONFIRMADO`), e segurar um celular/objeto para mostrar a caixa amarela de descarte (`OBJETO DETECTADO: IGNORADO`). |
| **12 a 15 min** | Todos | **Conclusão & Perguntas da Banca:** Apresentar as métricas de sustentabilidade ($kWh$, $R\$, CO_2$) e responder aos questionamentos dos professores. |

---

## 🛡️ FAQ: Como Responder às Perguntas Difíceis da Banca

1. **"Por que vocês não usaram OpenCV (`cv2.GaussianBlur` ou `cv2.Sobel`)?"**
   > *Resposta acadêmica:* "O item 3.a do Manual da APS da UNIP exige expressamente que os filtros sejam desenvolvidos pelo próprio grupo. Implementamos os laços espaciais de convolução discreta em Python puro com NumPy, respeitando os fundamentos teóricos de Gonzalez & Woods."
2. **"Como o sistema trata as bordas na convolução 2D?"**
   > *Resposta acadêmica:* "Implementamos tratamento de contorno por replicação de borda (padding por extensão de reflexão), evitando artefatos escuros nas extremidades da matriz."
3. **"Por que vocês usaram Abertura Morfológica e não Fechamento?"**
   > *Resposta acadêmica:* "A Abertura aplica erosão 3x3 seguida de dilatação 3x3. Sua função geométrica é eliminar ruídos pequenos isolados (ruído sal-e-pimenta de iluminação) sem alterar a massa volumétrica do ocupante."
4. **"Como vocês garantem que um celular ou objeto erguido não engane a rede neural?"**
   > *Resposta acadêmica:* "Utilizamos fusão em três níveis: delimitação da Região de Interesse (Bounding Box), inferência convolucional supervisionada sobre o recorte do alvo (e não no quadro global) e filtro fotométrico biométrico de pele humana (modelo de Kovac et al.), rejeitando qualquer superfície inanimada."

---

## ⚖️ Conformidade Rigorosa com o Manual da UNIP

| Exigência do Manual da APS | Implementação no VigIA | Arquivo no Projeto |
| :--- | :--- | :--- |
| **Item 3.a (Filtros de PDI desenvolvidos pelo grupo)** | Zero bibliotecas prontas. BT.601, Convolução 2D, Gauss 3x3, Sobel, Subtração e Abertura em código aberto. | [`prototipo_vigia.py`](prototipo_vigia.py) |
| **Item 3.b (Modelagem inteligente para imagens)** | Mini-CNN desenvolvida em Keras/TensorFlow com classificação binária (`vazio` / `ocupado`). | [`treinar_modelo.py`](treinar_modelo.py) |
| **Item 4 (Testes e evidências com fotos reais)** | As 6 etapas intermediárias de PDI salvas em alta resolução e mosaico 2x3 para a monografia. | [`etapas_processamento/`](etapas_processamento/) |
| **Cálculo de Sustentabilidade Ambiental** | Relatório de balanço energético em $kWh$, economia financeira em $R\$$ e redução de emissões em $kg \text{ de } CO_2$. | Módulo analítico integrado |
| **Suíte de Testes Automatizados** | 23 testes unitários e de integração cobrindo matemática de PDI, robustez e ciclo operacional E2E. | [`test_vigia.py`](test_vigia.py) |

---

## 📂 Estrutura de Arquivos do Repositório

```text
aps-6semestre/
│
├── README.md                   # Este guia completo e plano de ação do grupo
├── DOCUMENTACAO_APS_VIGIA.md   # Fundamentação teórica para o trabalho acadêmico
├── prototipo_vigia.py          # Núcleo de PDI artesanal e simulação de ciclos
├── interface_webcam.py         # Console visual ao vivo em 60 FPS com Pygame
├── treinar_modelo.py           # Script para re-treinar a mini-CNN Keras
├── test_vigia.py               # Suíte de testes automatizados (23 testes, 100% OK)
├── modelo_vigia.keras          # Modelo de rede neural convolucional serializado
│
├── dataset/                    # Amostras locais de treino (vazio e ocupado)
├── imagens_reais/              # Fotos reais de calibração e teste
├── etapas_processamento/       # As 6 evidências visuais salvas para a banca
└── documentos/                 # Manual da UNIP, slides em PPTX e PDF original
```
