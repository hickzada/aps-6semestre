"""
===================================================================================
PROJETO APS - CIÊNCIA DA COMPUTAÇÃO (5º/6º SEMESTRE) - UNIP
TEMA: Aplicação de Técnicas de Processamento Digital de Imagens e Visão 
      Computacional para Resolver Problemas de Sustentabilidade Ambiental.
SISTEMA: VigIA - Automação Inteligente de Iluminação por Visão Computacional
SUÍTE DE TESTES AUTOMATIZADOS END-TO-END (E2E) E GARANTIA DE QUALIDADE (QA)
===================================================================================
Conformidade Estrita com os Requisitos:
- R1: Processamento Digital de Imagens manual (sem OpenCV, BT.601, Sobel, Gauss, etc.)
- R2: Modelo de Visão Computacional Inteligente (mini-CNN Keras/TensorFlow)
- R3: Padrão formal de código e documentação acadêmica
- R4: Métricas de sustentabilidade (kWh, R$, CO2) e persistência de 6 etapas
- R5: Execução automatizada e determinística (python3 test_vigia.py ou pytest)
===================================================================================
"""

import math
import os
import sys
import unittest
import numpy as np
from PIL import Image

# Importação dos módulos do Sistema VigIA
import prototipo_vigia
from prototipo_vigia import ProcessadorImagem, DetectorPresencaVigIA, SistemaVigIA

# Importação opcional ou sob demanda do Keras/TensorFlow
try:
    import keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        keras = None


class TestPDIProcessamento(unittest.TestCase):
    """
    Camada 1: Validação analítica e algébrica dos algoritmos de Processamento
    Digital de Imagens (PDI) implementados manualmente em Python puro + NumPy/Pillow.
    Testa propriedades matemáticas, invariantes de domínio e condições de contorno.
    """

    def setUp(self):
        self.processador = ProcessadorImagem()

    def test_conversao_luminancia_bt601_cores_fundamentais(self):
        """
        Valida a conversão fotométrica ITU-R BT.601:
        Y = 0.299*R + 0.587*G + 0.114*B
        Testa as cores primárias analíticas e preservação em escala de cinza pura.
        """
        # 1. Branco puro (255, 255, 255) -> 255.0
        img_branco = Image.new("RGB", (2, 2), (255, 255, 255))
        cinza_branco = ProcessadorImagem.converter_para_tons_de_cinza(img_branco)
        self.assertAlmostEqual(cinza_branco[0][0], 255.0, places=3)

        # 2. Preto puro (0, 0, 0) -> 0.0
        img_preto = Image.new("RGB", (2, 2), (0, 0, 0))
        cinza_preto = ProcessadorImagem.converter_para_tons_de_cinza(img_preto)
        self.assertAlmostEqual(cinza_preto[0][0], 0.0, places=3)

        # 3. Vermelho puro (255, 0, 0) -> 0.299 * 255 = 76.245
        img_vermelho = Image.new("RGB", (2, 2), (255, 0, 0))
        cinza_vermelho = ProcessadorImagem.converter_para_tons_de_cinza(img_vermelho)
        self.assertAlmostEqual(cinza_vermelho[0][0], 76.245, places=3)

        # 4. Verde puro (0, 255, 0) -> 0.587 * 255 = 149.685
        img_verde = Image.new("RGB", (2, 2), (0, 255, 0))
        cinza_verde = ProcessadorImagem.converter_para_tons_de_cinza(img_verde)
        self.assertAlmostEqual(cinza_verde[0][0], 149.685, places=3)

        # 5. Azul puro (0, 0, 255) -> 0.114 * 255 = 29.070
        img_azul = Image.new("RGB", (2, 2), (0, 0, 255))
        cinza_azul = ProcessadorImagem.converter_para_tons_de_cinza(img_azul)
        self.assertAlmostEqual(cinza_azul[0][0], 29.070, places=3)

        # 6. Cinza uniforme (128, 128, 128) -> 128.0 (invariância monocromática)
        img_cinza = Image.new("RGB", (2, 2), (128, 128, 128))
        cinza_iso = ProcessadorImagem.converter_para_tons_de_cinza(img_cinza)
        self.assertAlmostEqual(cinza_iso[0][0], 128.0, places=3)

    def test_conversao_luminancia_bt601_faixa_dinamica(self):
        """
        Valida que qualquer combinação de cores RGB em [0, 255] gera
        luminância contida estritamente no intervalo normalizado [0.0, 255.0].
        """
        largura, altura = 8, 8
        dados_aleatorios = []
        for i in range(largura * altura):
            r = (i * 37) % 256
            g = (i * 73) % 256
            b = (i * 109) % 256
            dados_aleatorios.append((r, g, b))

        img = Image.new("RGB", (largura, altura))
        img.putdata(dados_aleatorios)
        matriz_cinza = ProcessadorImagem.converter_para_tons_de_cinza(img)

        self.assertEqual(len(matriz_cinza), altura)
        self.assertEqual(len(matriz_cinza[0]), largura)

        for linha in matriz_cinza:
            for val in linha:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 255.0)

    def test_convolucao_2d_identidade_e_preservacao_dimensional(self):
        """
        Valida o algoritmo manual de convolução 2D discreta:
        - Preservação exata das dimensões H x W da matriz original.
        - Resposta a kernel identidade delta com elemento central unitário.
        """
        altura, largura = 6, 8
        matriz_entrada = [[float(y * 10 + x) for x in range(largura)] for y in range(altura)]

        kernel_identidade = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ]

        matriz_saida = ProcessadorImagem.aplicar_convolucao_2d(
            matriz_entrada, kernel_identidade, divisor=1.0, deslocamento=0.0
        )

        self.assertEqual(len(matriz_saida), altura)
        self.assertEqual(len(matriz_saida[0]), largura)

        for y in range(altura):
            for x in range(largura):
                self.assertAlmostEqual(matriz_saida[y][x], matriz_entrada[y][x], places=4)

    def test_convolucao_2d_campo_constante_e_padding_extensao(self):
        """
        Valida a condição de contorno por replicação/extensão de borda (clamping):
        Em um campo homogêneo constante C com kernel normalizado (soma dos pesos = divisor),
        todos os pixels de saída, inclusive bordas e cantos extremos, devem preservar C.
        """
        altura, largura = 7, 7
        valor_constante = 150.0
        matriz_homogenea = [[valor_constante for _ in range(largura)] for _ in range(altura)]

        # Kernel Gaussiano 3x3 normalizado por divisor 16
        mascara_gauss = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]

        matriz_filtrada = ProcessadorImagem.aplicar_convolucao_2d(
            matriz_homogenea, mascara_gauss, divisor=16.0
        )

        # Testa cantos extremos onde o padding atua duplamente
        cantos = [(0, 0), (0, largura - 1), (altura - 1, 0), (altura - 1, largura - 1)]
        for cy, cx in cantos:
            self.assertAlmostEqual(matriz_filtrada[cy][cx], valor_constante, places=4,
                                   msg=f"Falha de padding de borda no canto ({cy}, {cx})")

        # Testa integralidade da matriz
        for y in range(altura):
            for x in range(largura):
                self.assertAlmostEqual(matriz_filtrada[y][x], valor_constante, places=4)

    def test_filtro_gaussiano_resposta_ao_impulso_unitario(self):
        """
        Valida a função de espalhamento pontual (PSF) do filtro Gaussiano 3x3.
        Submete uma função delta de Dirac (pixel central = 255.0 em matriz 5x5 de zeros).
        Os coeficientes resultantes devem ser exatamente proporcionais aos pesos gaussianos:
        Centro: 255 * 4/16 = 63.75
        Ortogonais: 255 * 2/16 = 31.875
        Diagonais: 255 * 1/16 = 15.9375
        Externos: 0.0
        """
        matriz_delta = [[0.0 for _ in range(5)] for _ in range(5)]
        matriz_delta[2][2] = 255.0

        suave = ProcessadorImagem.filtro_suavizacao_gaussiana_3x3(matriz_delta)

        # Centro (2, 2)
        self.assertAlmostEqual(suave[2][2], 63.75, places=3)

        # Vizinhos ortogonais (norte, sul, leste, oeste)
        for oy, ox in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            self.assertAlmostEqual(suave[oy][ox], 31.875, places=3)

        # Vizinhos diagonais
        for dy, dx in [(1, 1), (1, 3), (3, 1), (3, 3)]:
            self.assertAlmostEqual(suave[dy][dx], 15.9375, places=3)

        # Pixels distantes (fora do suporte 3x3)
        for fy, fx in [(0, 0), (0, 4), (4, 0), (4, 4)]:
            self.assertAlmostEqual(suave[fy][fx], 0.0, places=3)

    def test_filtro_gaussiano_reducao_de_variancia_ruido(self):
        """
        Valida o efeito de atenuação de ruído estocástico de alta frequência:
        A variância amostral de uma matriz com ruído deve ser estritamente menor
        após a aplicação do filtro passa-baixas Gaussiano.
        """
        matriz_ruidosa = [
            [0.0, 255.0, 0.0, 255.0, 0.0, 255.0],
            [255.0, 0.0, 255.0, 0.0, 255.0, 0.0],
            [0.0, 255.0, 0.0, 255.0, 0.0, 255.0],
            [255.0, 0.0, 255.0, 0.0, 255.0, 0.0],
            [0.0, 255.0, 0.0, 255.0, 0.0, 255.0],
            [255.0, 0.0, 255.0, 0.0, 255.0, 0.0],
        ]

        var_original = float(np.var(matriz_ruidosa))
        matriz_suave = ProcessadorImagem.filtro_suavizacao_gaussiana_3x3(matriz_ruidosa)
        var_suave = float(np.var(matriz_suave))

        self.assertLess(var_suave, var_original,
                        "O filtro gaussiano deve reduzir a variância amostral de alta frequência.")

    def test_filtro_sobel_campo_homogeneo_e_deteccao_arestas(self):
        """
        Valida o operador diferencial de Sobel (passa-altas):
        - Em campo homogêneo sem variação de gradiente, a magnitude deve ser 0.0 em toda a extensão.
        - Em degrau vertical abrupto, Gx detecta a borda e Gy permanece nulo na interface.
        - Em degrau horizontal abrupto, Gy detecta a borda e Gx permanece nulo na interface.
        """
        # 1. Campo Homogêneo
        matriz_plana = [[120.0 for _ in range(6)] for _ in range(6)]
        sobel_plano = ProcessadorImagem.filtro_realce_bordas_sobel(matriz_plana)
        for linha in sobel_plano:
            for val in linha:
                self.assertAlmostEqual(val, 0.0, places=3,
                                       msg="Gradiente de Sobel em campo plano deve ser estritamente nulo.")

        # 2. Degrau Vertical (metade esquerda = 0, metade direita = 100)
        matriz_vert = [[0.0, 0.0, 0.0, 100.0, 100.0, 100.0] for _ in range(6)]
        sobel_vert = ProcessadorImagem.filtro_realce_bordas_sobel(matriz_vert)
        # Ao longo da coluna divisória (x=2 ou x=3), a magnitude deve ser estritamente positiva
        for y in range(1, 5):
            self.assertGreater(sobel_vert[y][2], 0.0)
            self.assertGreater(sobel_vert[y][3], 0.0)

        # 3. Degrau Horizontal (metade superior = 0, metade inferior = 100)
        matriz_horiz = [[0.0] * 6 for _ in range(3)] + [[100.0] * 6 for _ in range(3)]
        sobel_horiz = ProcessadorImagem.filtro_realce_bordas_sobel(matriz_horiz)
        for x in range(1, 5):
            self.assertGreater(sobel_horiz[2][x], 0.0)
            self.assertGreater(sobel_horiz[3][x], 0.0)

    def test_subtracao_plano_de_fundo_propriedades_metricas(self):
        """
        Valida as propriedades métricas do operador de subtração de fundo:
        Delta(x, y) = |Atual(x, y) - Fundo(x, y)|
        - Identidade dos indiscerníveis: |A - A| = 0
        - Simetria: |A - B| = |B - A|
        - Desigualdade triangular: |A - C| <= |A - B| + |B - C|
        """
        A = [[50.0, 80.0], [120.0, 200.0]]
        B = [[40.0, 95.0], [100.0, 210.0]]
        C = [[70.0, 60.0], [130.0, 190.0]]

        # Identidade
        diff_aa = ProcessadorImagem.subtrair_imagens(A, A)
        for linha in diff_aa:
            for val in linha:
                self.assertEqual(val, 0.0)

        # Simetria
        diff_ab = ProcessadorImagem.subtrair_imagens(A, B)
        diff_ba = ProcessadorImagem.subtrair_imagens(B, A)
        for y in range(2):
            for x in range(2):
                self.assertAlmostEqual(diff_ab[y][x], diff_ba[y][x], places=4)

        # Desigualdade triangular
        diff_ac = ProcessadorImagem.subtrair_imagens(A, C)
        diff_bc = ProcessadorImagem.subtrair_imagens(B, C)
        for y in range(2):
            for x in range(2):
                self.assertLessEqual(diff_ac[y][x], diff_ab[y][x] + diff_bc[y][x] + 1e-6)

    def test_limiarizacao_binaria_rigida(self):
        """
        Valida o operador de binarização (Thresholding global rígido):
        Se pixel >= limiar -> 255
        Se pixel < limiar -> 0
        Verifica o ponto de transição estrito para limiar = 28.
        """
        matriz_teste = [
            [0.0, 15.0, 27.0, 27.9],
            [28.0, 28.1, 50.0, 255.0]
        ]
        limiar = 28.0
        binaria = ProcessadorImagem.aplicar_limiarizacao(matriz_teste, limiar=limiar)

        # Valores abaixo do limiar devem ser mapeados para 0
        self.assertEqual(binaria[0][0], 0)
        self.assertEqual(binaria[0][1], 0)
        self.assertEqual(binaria[0][2], 0)
        self.assertEqual(binaria[0][3], 0)

        # Valores a partir do limiar devem ser mapeados para 255
        self.assertEqual(binaria[1][0], 255)
        self.assertEqual(binaria[1][1], 255)
        self.assertEqual(binaria[1][2], 255)
        self.assertEqual(binaria[1][3], 255)

        # Faixa estrita binária {0, 255}
        for linha in binaria:
            for val in linha:
                self.assertIn(val, (0, 255))

    def test_morfologia_matematica_abertura_3x3(self):
        """
        Valida o operador de Abertura Morfológica artesanal (Erosão 3x3 seguida de Dilatação 3x3):
        - Eliminação estrita de ruído impulsivo isolado (pixel 255 cercado por zeros).
        - Preservação do núcleo de objetos conectados maiores que o elemento estruturante 3x3.
        - Antiextensividade: a abertura nunca insere pixels ativos onde a entrada era nula.
        """
        # 1. Ruído Isolado (pixel branco solitário em reticulado 7x7 de zeros)
        matriz_ruido = [[0 for _ in range(7)] for _ in range(7)]
        matriz_ruido[3][3] = 255

        aberta_ruido = ProcessadorImagem.filtro_morfologico_abertura(matriz_ruido)
        self.assertEqual(aberta_ruido[3][3], 0, "Pixel isolado deve ser eliminado pela abertura morfológica.")
        for linha in aberta_ruido:
            for val in linha:
                self.assertEqual(val, 0)

        # 2. Estrutura Conexa (bloco 5x5 de pixels 255 centralizado em matriz 9x9)
        matriz_bloco = [[0 for _ in range(9)] for _ in range(9)]
        for y in range(2, 7):
            for x in range(2, 7):
                matriz_bloco[y][x] = 255

        aberta_bloco = ProcessadorImagem.filtro_morfologico_abertura(matriz_bloco)

        # O núcleo central 3x3 (linhas 3 a 5, colunas 3 a 5) deve ser preservado com valor 255
        for y in range(3, 6):
            for x in range(3, 6):
                self.assertEqual(aberta_bloco[y][x], 255,
                                 f"Núcleo conexo na posição ({y}, {x}) deve permanecer ativo.")

        # 3. Antiextensividade
        for y in range(9):
            for x in range(9):
                if matriz_bloco[y][x] == 0:
                    self.assertEqual(aberta_bloco[y][x], 0,
                                     "Operador antiextensivo não pode criar pixels onde a entrada era zero.")


class TestModeloInteligente(unittest.TestCase):
    """
    Camada 2: Validação da Rede Neural Convolucional (mini-CNN Keras/TensorFlow).
    Verifica o artefato compilado 'modelo_vigia.keras', compatibilidade tensorial
    de entrada (batch, 64, 64, 3) e predição no intervalo fechado [0.0, 1.0].
    """

    @classmethod
    def setUpClass(cls):
        """
        Assegura a disponibilidade do artefato modelo_vigia.keras para os testes.
        Caso o arquivo não exista mas treinar_modelo.py esteja presente, executa o treino.
        """
        cls.caminho_modelo = "modelo_vigia.keras"
        if not os.path.exists(cls.caminho_modelo):
            if os.path.exists("treinar_modelo.py"):
                import treinar_modelo
                treinar_modelo.treinar_modelo_vigia(epocas=5)

    def test_presenca_e_carregamento_artefato_keras(self):
        """
        Valida que o artefato serializado modelo_vigia.keras existe em disco,
        possui tamanho representativo de uma CNN compilada e carrega com sucesso via Keras.
        """
        self.assertTrue(os.path.exists(self.caminho_modelo),
                        f"O arquivo serializado '{self.caminho_modelo}' deve existir na raiz do projeto.")
        tamanho_bytes = os.path.getsize(self.caminho_modelo)
        self.assertGreater(tamanho_bytes, 1024,
                           "O tamanho do artefato compilado Keras deve ser superior a 1 KB.")

        self.assertIsNotNone(keras, "TensorFlow/Keras deve estar instalado e importável no ambiente.")
        modelo = keras.models.load_model(self.caminho_modelo)
        self.assertIsNotNone(modelo, "A função keras.models.load_model deve desserializar o modelo sem exceções.")

    def test_arquitetura_e_shape_camadas(self):
        """
        Valida a topologia hierárquica da mini-CNN:
        - Tensor de entrada compatível com resolução espacial (64, 64, 3).
        - Camada de saída com exatamente 1 neurônio e ativação sigmoide logística.
        - Presença de camadas convolucionais 2D e subamostragem max-pooling.
        """
        modelo = keras.models.load_model(self.caminho_modelo)

        # Formato de entrada esperado
        self.assertEqual(modelo.input_shape, (None, 64, 64, 3),
                         "O formato tensorial de entrada deve ser (batch, 64, 64, 3).")

        # Camada de saída
        camada_saida = modelo.layers[-1]
        self.assertEqual(camada_saida.units, 1,
                         "A camada de saída deve possuir 1 neurônio para classificação binária.")
        nome_ativacao = getattr(camada_saida.activation, "__name__", str(camada_saida.activation))
        self.assertIn("sigmoid", nome_ativacao.lower(),
                      "A função de ativação da camada de saída deve ser sigmoide.")

        # Presença de blocos convolucionais
        nomes_camadas = [layer.__class__.__name__ for layer in modelo.layers]
        self.assertIn("Conv2D", nomes_camadas, "A arquitetura deve conter camadas convolucionais Conv2D.")
        self.assertIn("MaxPooling2D", nomes_camadas, "A arquitetura deve conter camadas de MaxPooling2D.")

    def test_predicao_probabilistica_intervalo_unitario(self):
        """
        Valida que o modelo executa a inferência sobre tensores de teste
        e retorna uma probabilidade escalar contida estritamente no intervalo [0.0, 1.0].
        """
        modelo = keras.models.load_model(self.caminho_modelo)

        tensor_dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
        predicao = modelo.predict(tensor_dummy, verbose=0)

        self.assertEqual(predicao.shape, (1, 1),
                         "A predição deve retornar formato tensorial (1, 1).")
        probabilidade = float(predicao[0][0])
        self.assertGreaterEqual(probabilidade, 0.0,
                                "A probabilidade inferida deve ser maior ou igual a 0.0.")
        self.assertLessEqual(probabilidade, 1.0,
                             "A probabilidade inferida deve ser menor ou igual a 1.0.")

    def test_inferencia_amostras_reais_discriminacao(self):
        """
        Valida que a mini-CNN discrimina adequadamente as fotos reais do projeto:
        A probabilidade inferida para 'escritorio_com_pessoas.jpg' deve ser
        superior à inferida para 'escritorio_vazio.jpg'.
        """
        caminho_ocupado = os.path.join("imagens_reais", "escritorio_com_pessoas.jpg")
        caminho_vazio = os.path.join("imagens_reais", "escritorio_vazio.jpg")

        self.assertTrue(os.path.exists(caminho_ocupado), "Foto real com pessoas deve existir.")
        self.assertTrue(os.path.exists(caminho_vazio), "Foto real de sala vazia deve existir.")

        modelo = keras.models.load_model(self.caminho_modelo)

        img_ocupada = Image.open(caminho_ocupado).convert("RGB").resize((64, 64))
        img_vazia = Image.open(caminho_vazio).convert("RGB").resize((64, 64))

        arr_ocupado = np.expand_dims(np.array(img_ocupada, dtype=np.float32), axis=0)
        arr_vazio = np.expand_dims(np.array(img_vazia, dtype=np.float32), axis=0)

        prob_ocupado = float(modelo.predict(arr_ocupado, verbose=0)[0][0])
        prob_vazio = float(modelo.predict(arr_vazio, verbose=0)[0][0])

        self.assertGreater(prob_ocupado, prob_vazio,
                           f"A probabilidade de presença na imagem ocupada ({prob_ocupado:.3f}) "
                           f"deve ser estritamente superior à imagem vazia ({prob_vazio:.3f}).")


class TestSistemaVigIAE2E(unittest.TestCase):
    """
    Camada 3: Validação Fim a Fim (End-to-End) do Sistema VigIA Integrado.
    - Ciclos operacionais prediais com fotografias reais.
    - Decisão de controle de carga de iluminação (LIGADA / DESLIGADA).
    - Persistência e integridade dimensional das 6 etapas intermediárias de PDI.
    - Verificação analítica do balanço físico-financeiro de sustentabilidade.
    """

    def setUp(self):
        self.caminho_vazio = os.path.join("imagens_reais", "escritorio_vazio.jpg")
        self.caminho_ocupado = os.path.join("imagens_reais", "escritorio_com_pessoas.jpg")
        self.pasta_etapas = "etapas_processamento"
        os.makedirs(self.pasta_etapas, exist_ok=True)

    def test_ciclo_operacional_sala_ocupada_aciona_luz(self):
        """
        Cenário: Imagem real com pessoas e sinal PIR ativo.
        Comportamento esperado:
        - Presença humana confirmada (presenca = True).
        - Circuito de iluminação comutado/mantido para LIGADA (luz_ligada = True).
        - Confiança percentual estatística >= 50.0%.
        """
        sistema = SistemaVigIA(potencia_iluminacao_watts=400, tarifa_kwh=0.85)
        sistema.calibrar_fundo(self.caminho_vazio)

        resultado = sistema.processar_ciclo(
            caminho_imagem_atual=self.caminho_ocupado,
            sinal_sensor_pir=True,
            duracao_minutos=60
        )

        self.assertIsInstance(resultado, dict)
        self.assertTrue(resultado["luz_ligada"],
                        "O sistema DEVE manter a iluminação LIGADA quando há ocupação humana.")
        self.assertTrue(resultado["presenca"],
                        "O sistema DEVE classificar presença humana confirmada na cena ocupada.")

        confianca = resultado.get("confianca", resultado.get("confianca_percentual", 0.0))
        self.assertGreaterEqual(confianca, 50.0,
                                "A confiança estatística para sala ocupada deve ser superior a 50%.")

    def test_ciclo_operacional_sala_vazia_desliga_luz(self):
        """
        Cenário: Imagem real de escritório vazio e PIR inativo após expediente.
        Comportamento esperado:
        - Ausência humana detectada (presenca = False).
        - Iluminação desativada para corte de desperdício (luz_ligada = False).
        - Acumulação correta do tempo ocioso para sustentabilidade.
        """
        sistema = SistemaVigIA(potencia_iluminacao_watts=400, tarifa_kwh=0.85)
        sistema.calibrar_fundo(self.caminho_vazio)

        # Estado inicial com luz acesa
        sistema.luz_acesa = True

        resultado = sistema.processar_ciclo(
            caminho_imagem_atual=self.caminho_vazio,
            sinal_sensor_pir=False,
            duracao_minutos=30
        )

        self.assertFalse(resultado["luz_ligada"],
                         "O sistema DEVE desligar a iluminação na ausência de ocupantes.")
        self.assertFalse(resultado["presenca"],
                         "O sistema NÃO deve classificar presença humana na sala desocupada.")
        self.assertAlmostEqual(sistema.horas_luz_economizada, 0.5, places=2,
                               msg="O tempo de luz economizada deve registrar 0.5 horas (30 minutos).")

    def test_persistencia_seis_etapas_intermediarias_pdi(self):
        """
        Valida a persistência física em disco e conformidade dimensional das 6 etapas
        intermediárias de Processamento Digital de Imagens exigidas na monografia da APS:
        1. 1_tons_de_cinza.png
        2. 2_suavizacao_gaussiana.png
        3. 3_realce_sobel.png
        4. 4_subtracao_fundo.png
        5. 5_binarizacao_limiar.png
        6. 6_morfologia_final.png
        """
        sistema = SistemaVigIA(potencia_iluminacao_watts=400, tarifa_kwh=0.85)
        sistema.calibrar_fundo(self.caminho_vazio)

        # Executa ciclo com exportação de debug
        sistema.processar_ciclo(
            caminho_imagem_atual=self.caminho_ocupado,
            sinal_sensor_pir=True,
            duracao_minutos=15,
            pasta_saida_debug=self.pasta_etapas
        )

        arquivos_esperados = [
            "1_tons_de_cinza.png",
            "2_suavizacao_gaussiana.png",
            "3_realce_sobel.png",
            "4_subtracao_fundo.png",
            "5_binarizacao_limiar.png",
            "6_morfologia_final.png"
        ]

        for nome_arquivo in arquivos_esperados:
            caminho_arquivo = os.path.join(self.pasta_etapas, nome_arquivo)
            self.assertTrue(os.path.exists(caminho_arquivo),
                            f"A evidência visual '{nome_arquivo}' deve existir em disco.")
            self.assertGreater(os.path.getsize(caminho_arquivo), 0,
                               f"O arquivo '{nome_arquivo}' não pode estar vazio (tamanho > 0).")

            with Image.open(caminho_arquivo) as img:
                self.assertEqual(img.size, (320, 240),
                                 f"A evidência '{nome_arquivo}' deve possuir resolução exata de 320x240 pixels.")
                self.assertEqual(img.mode, "L",
                                 f"A evidência '{nome_arquivo}' deve ser salva em escala de cinza de 8 bits (modo 'L').")

    def test_metricas_sustentabilidade_balanco_energetico_e_carbono(self):
        """
        Valida as equações do modelo físico de sustentabilidade:
        - Consumo Convencional: E_base = (P * t_tot) / 1000 [kWh]
        - Consumo VigIA: E_vigia = (P * (t_tot - t_eco)) / 1000 [kWh]
        - Energia Poupada: Delta_E = (P * t_eco) / 1000 [kWh]
        - Economia Financeira: S_fin = Delta_E * C_tarifa [R$]
        - Redução de Pegada de Carbono: M_co2 = Delta_E * 0.084 [kg CO2]
        - Percentual de Conservação: (Delta_E / E_base) * 100 [%]
        """
        potencia = 400.0  # Watts
        tarifa = 0.85     # R$/kWh
        fator_co2 = 0.084 # kg CO2/kWh

        sistema = SistemaVigIA(potencia_iluminacao_watts=potencia, tarifa_kwh=tarifa)
        sistema.calibrar_fundo(self.caminho_vazio)

        # Ciclo 1: 60 minutos ocupado (luz ligada, 0 economia)
        sistema.processar_ciclo(self.caminho_ocupado, sinal_sensor_pir=True, duracao_minutos=60)

        # Ciclo 2: 60 minutos desocupado (luz desligada, 1 hora poupada)
        sistema.processar_ciclo(self.caminho_vazio, sinal_sensor_pir=False, duracao_minutos=60)

        # Ciclo 3: 60 minutos sentinela em guarda (luz permanece desligada, 1 hora poupada)
        sistema.processar_ciclo(self.caminho_vazio, sinal_sensor_pir=False, duracao_minutos=60)

        t_tot = 3.0 # horas
        t_eco = 2.0 # horas

        self.assertAlmostEqual(sistema.horas_totais_monitoradas, t_tot, places=2)
        self.assertAlmostEqual(sistema.horas_luz_economizada, t_eco, places=2)

        kwh_conv_esp = (potencia * t_tot) / 1000.0
        kwh_vigia_esp = (potencia * (t_tot - t_eco)) / 1000.0
        kwh_poup_esp = (potencia * t_eco) / 1000.0
        fin_esp = kwh_poup_esp * tarifa
        co2_esp = kwh_poup_esp * fator_co2
        pct_esp = (kwh_poup_esp / kwh_conv_esp) * 100.0

        # Validação do método estruturado de dicionário (Interface Contract M3)
        if hasattr(sistema, "obter_relatorio_sustentabilidade"):
            metricas = sistema.obter_relatorio_sustentabilidade()
            self.assertIsInstance(metricas, dict)
            self.assertAlmostEqual(metricas["kwh_convencional"], kwh_conv_esp, places=3)
            self.assertAlmostEqual(metricas["kwh_vigia"], kwh_vigia_esp, places=3)
            self.assertAlmostEqual(metricas["kwh_poupados"], kwh_poup_esp, places=3)
            self.assertAlmostEqual(metricas["economia_reais"], fin_esp, places=2)
            self.assertAlmostEqual(metricas["reducao_co2_kg"], co2_esp, places=3)
            self.assertAlmostEqual(metricas["percentual_economia"], pct_esp, places=2)

        # Validação do relatório textual acadêmico
        relatorio_texto = sistema.gerar_relatorio_sustentabilidade()
        self.assertIn("RELATÓRIO DE SUSTENTABILIDADE", relatorio_texto)
        self.assertIn("ENERGIA TOTAL POUPADA", relatorio_texto)
        self.assertIn("ECONOMIA FINANCEIRA ESTIMADA", relatorio_texto)
        self.assertIn("REDUÇÃO DE PEGADA DE CARBONO", relatorio_texto)


class TestRobustezTratamentoErros(unittest.TestCase):
    """
    Camada 4: Testes de Robustez, Casos de Borda e Tolerância a Falhas Excepcionais.
    Garante resiliência operacional contra anomalias e divisões por zero.
    """

    def test_excecao_sistema_nao_calibrado(self):
        """
        Valida que a tentativa de processar um ciclo sem calibração prévia
        de fundo lança explicitamente RuntimeError com mensagem instrutiva.
        """
        sistema = SistemaVigIA()
        with self.assertRaises(RuntimeError):
            sistema.processar_ciclo("imagens_reais/escritorio_com_pessoas.jpg")

    def test_excecao_arquivo_imagem_inexistente(self):
        """
        Valida que o carregador de imagens lança FileNotFoundError caso
        o caminho especificado não aponte para um arquivo existente em disco.
        """
        with self.assertRaises(FileNotFoundError):
            ProcessadorImagem.carregar_imagem("arquivo_inexistente_de_teste_12345.jpg")

    def test_divisao_segura_duracao_zero_e_pixels_zero(self):
        """
        Valida que durações nulas e ausência total de pixels ativos (diferença nula)
        não causam ZeroDivisionError e mantêm estados consistentes.
        """
        sistema = SistemaVigIA()
        sistema.calibrar_fundo("imagens_reais/escritorio_vazio.jpg")

        # Ciclo com duração 0 minutos
        resultado = sistema.processar_ciclo("imagens_reais/escritorio_vazio.jpg",
                                            sinal_sensor_pir=False,
                                            duracao_minutos=0)
        self.assertAlmostEqual(sistema.horas_totais_monitoradas, 0.0, places=4)

        # Detecção com matrizes perfeitamente idênticas (0 pixels ativos)
        matriz_suave = [[100.0] * 10 for _ in range(10)]
        detector = DetectorPresencaVigIA()
        relatorio = detector.analisar_quadro(matriz_suave, matriz_suave)

        self.assertEqual(relatorio["pixels_ativos"], 0)
        self.assertIsNone(relatorio["centroide"],
                          "O centroide deve ser None quando não há pixels ativos (sem divisão por zero).")

    def test_estabilidade_matrizes_extremas_preto_e_branco(self):
        """
        Valida que matrizes homogêneas extremas (100% preta com valor 0.0 ou
        100% branca saturada com valor 255.0) transitam pelos filtros de convolução,
        Gaussiano, Sobel e Morfologia sem estouro de pilha, exceção aritmética ou overflow.
        """
        for valor in [0.0, 255.0]:
            matriz_extrema = [[valor for _ in range(5)] for _ in range(5)]

            # Convolução e Gaussiano
            suave = ProcessadorImagem.filtro_suavizacao_gaussiana_3x3(matriz_extrema)
            for linha in suave:
                for v in linha:
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 255.0)

            # Sobel
            sobel = ProcessadorImagem.filtro_realce_bordas_sobel(matriz_extrema)
            for linha in sobel:
                for v in linha:
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 255.0)

            # Morfologia
            matriz_bin = [[int(valor) for _ in range(5)] for _ in range(5)]
            morf = ProcessadorImagem.filtro_morfologico_abertura(matriz_bin)
            self.assertEqual(len(morf), 5)
            self.assertEqual(len(morf[0]), 5)

    def test_comportamento_modo_sentinela_pir(self):
        """
        Valida o Modo Sentinela do VigIA:
        Quando a lâmpada já está desligada e o PIR está inativo, o sistema
        pula o processamento matricial da câmera, mantém as lâmpadas desligadas
        e acumula o tempo de corte de iluminação.
        """
        sistema = SistemaVigIA()
        sistema.calibrar_fundo("imagens_reais/escritorio_vazio.jpg")
        sistema.luz_acesa = False

        resultado = sistema.processar_ciclo(
            caminho_imagem_atual="imagens_reais/escritorio_vazio.jpg",
            sinal_sensor_pir=False,
            duracao_minutos=45
        )

        self.assertFalse(resultado["luz_ligada"])
        self.assertEqual(resultado["motivo"], "Sentinela PIR em guarda")
        self.assertAlmostEqual(sistema.horas_luz_economizada, 0.75, places=2)


if __name__ == "__main__":
    # Permite execução direta via 'python3 test_vigia.py' com relatório verboso
    unittest.main(verbosity=2)
