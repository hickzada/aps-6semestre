"""
===================================================================================
PROJETO APS - CIÊNCIA DA COMPUTAÇÃO (5º/6º SEMESTRE) - UNIP
TEMA: Aplicação de Técnicas de Processamento Digital de Imagens e Visão 
      Computacional para Resolver Problemas de Sustentabilidade Ambiental.
SISTEMA: VigIA - Automação Inteligente de Iluminação por Visão Computacional
===================================================================================

CONFORMIDADE COM O MANUAL DA APS:
1. "Os filtros devem ser desenvolvidos pelo grupo preferencialmente em Python":
   - Convolução 2D implementada do zero com tratamento de borda (padding por extensão).
   - Conversão de RGB para Tons de Cinza pela fórmula de luminância ITU-R BT.601.
   - Filtro Passa-Baixas Gaussiano 3x3 manual para redução de ruído.
   - Filtro Passa-Altas Sobel (Gx e Gy) manual para realce de bordas.
   - Segmentação por Subtração de Plano de Fundo pixel a pixel.
   - Binarização (Limiarização) manual rígida.
   - Operação Morfológica de Abertura (Erosão + Dilatação) manual contra ruído.
2. Uso de Imagens Reais:
   - Suporta fotos reais de câmeras CFTV e salas de aula.
   - Redimensionamento automático na aquisição para viabilizar processamento ágil.
3. Todas as variáveis e funções em português e detalhadamente comentadas.
===================================================================================
"""

import math
import os
import sys
import numpy as np
from PIL import Image


# ===================================================================================
# 1. MÓDULO DE PROCESSAMENTO DIGITAL DE IMAGENS (PDI MANUAL)
# ===================================================================================

class ProcessadorImagem:
    """
    Módulo responsável por todas as transformações de PDI exigidas no manual.
    Nenhuma função pronta de convolução/filtros de terceiros (ex: OpenCV) é usada.
    """

    @staticmethod
    def carregar_imagem(caminho_arquivo, redimensionar=(320, 240)):
        """
        Carrega uma imagem real do disco ou recebe instancia PIL.Image e converte para RGB.
        Aplica redimensionamento proporcional para resolução padrão de processamento
        (padrão da indústria de CFTV para garantir velocidade em tempo real).
        """
        if isinstance(caminho_arquivo, Image.Image):
            imagem = caminho_arquivo.convert("RGB")
        else:
            if not os.path.exists(caminho_arquivo):
                raise FileNotFoundError(f"Arquivo de imagem não encontrado: {caminho_arquivo}")
            imagem = Image.open(caminho_arquivo).convert("RGB")

        if redimensionar is not None:
            # Redimensionamento suave para processamento eficiente
            imagem = imagem.resize(redimensionar, Image.Resampling.BILINEAR)

        return imagem

    @staticmethod
    def salvar_imagem(matriz_pixels, caminho_saida, modo="L"):
        """
        Converte uma matriz bidimensional de volta em imagem e salva no disco.
        modo 'L' = Escala de Cinza (0 a 255)
        """
        altura = len(matriz_pixels)
        largura = len(matriz_pixels[0])

        nova_imagem = Image.new(modo, (largura, altura))
        pixels_planificados = []

        for y in range(altura):
            for x in range(largura):
                # Clamping estrito para garantir intervalo válido de 8 bits [0, 255]
                valor = int(max(0, min(255, round(matriz_pixels[y][x]))))
                pixels_planificados.append(valor)

        nova_imagem.putdata(pixels_planificados)
        nova_imagem.save(caminho_saida)
        return caminho_saida

    @staticmethod
    def converter_para_tons_de_cinza(imagem_pil):
        """
        Fórmula de Luminância ITU-R BT.601:
        Y = 0.299*R + 0.587*G + 0.114*B
        Simula a sensibilidade do olho humano, que é mais sensível ao canal verde.
        """
        largura, altura = imagem_pil.size
        if hasattr(imagem_pil, "get_flattened_data"):
            pixels = list(imagem_pil.get_flattened_data())
        else:
            pixels = list(imagem_pil.getdata())

        matriz_cinza = []
        for y in range(altura):
            linha = []
            for x in range(largura):
                r, g, b = pixels[y * largura + x]
                luminancia = (0.299 * r) + (0.587 * g) + (0.114 * b)
                linha.append(luminancia)
            matriz_cinza.append(linha)

        return matriz_cinza

    @staticmethod
    def aplicar_convolucao_2d(matriz_cinza, mascara_kernel, divisor=1.0, deslocamento=0.0):
        """
        Algoritmo central de convolução bidimensional discreta desenvolvido pelo grupo.
        Aplica uma máscara matricial (kernel) sobre cada pixel da imagem.
        Tratamento de borda: replicação dos pixels da borda (Padding por extensão).
        """
        altura = len(matriz_cinza)
        largura = len(matriz_cinza[0])

        tamanho_kernel = len(mascara_kernel)
        raio = tamanho_kernel // 2

        matriz_resultado = []

        for y in range(altura):
            linha_resultado = []
            for x in range(largura):
                soma = 0.0

                for ky in range(tamanho_kernel):
                    for kx in range(tamanho_kernel):
                        # Deslocamento espacial relativo ao centro do kernel
                        pos_y = y + (ky - raio)
                        pos_x = x + (kx - raio)

                        # Tratamento de borda com espelhamento/extensão
                        pos_y = max(0, min(altura - 1, pos_y))
                        pos_x = max(0, min(largura - 1, pos_x))

                        peso = mascara_kernel[ky][kx]
                        valor_pixel = matriz_cinza[pos_y][pos_x]
                        soma += valor_pixel * peso

                valor_final = (soma / divisor) + deslocamento
                valor_final = max(0.0, min(255.0, valor_final))
                linha_resultado.append(valor_final)

            matriz_resultado.append(linha_resultado)

        return matriz_resultado

    @classmethod
    def filtro_suavizacao_gaussiana_3x3(cls, matriz_cinza):
        """
        Filtro de suavização espacial (passa-baixas) Gaussiano 3x3.
        Atenua ruídos de alta frequência e granulação de sensores de CFTV.
        Kernel:
        [1, 2, 1]
        [2, 4, 2] * (1/16)
        [1, 2, 1]
        """
        mascara_gaussiana = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]
        return cls.aplicar_convolucao_2d(matriz_cinza, mascara_gaussiana, divisor=16.0)

    @classmethod
    def filtro_realce_bordas_sobel(cls, matriz_cinza):
        """
        Filtro gradiente de Sobel (passa-altas) para realce de contornos.
        Calcula o gradiente horizontal (Gx) e vertical (Gy) através de convolução manual
        e computa a magnitude euclidiana da borda: Magnitude = sqrt(Gx^2 + Gy^2).
        """
        mascara_gx = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]

        mascara_gy = [
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]

        altura = len(matriz_cinza)
        largura = len(matriz_cinza[0])
        raio = 1

        matriz_sobel = []

        for y in range(altura):
            linha = []
            for x in range(largura):
                soma_gx = 0.0
                soma_gy = 0.0

                for ky in range(3):
                    for kx in range(3):
                        pos_y = max(0, min(altura - 1, y + (ky - raio)))
                        pos_x = max(0, min(largura - 1, x + (kx - raio)))
                        pixel = matriz_cinza[pos_y][pos_x]

                        soma_gx += pixel * mascara_gx[ky][kx]
                        soma_gy += pixel * mascara_gy[ky][kx]

                magnitude = math.sqrt((soma_gx ** 2) + (soma_gy ** 2))
                magnitude = max(0.0, min(255.0, magnitude))
                linha.append(magnitude)

            matriz_sobel.append(linha)

        return matriz_sobel

    @staticmethod
    def subtrair_imagens(matriz_atual, matriz_fundo_referencia):
        """
        Subtração de fundo pixel a pixel:
        Diferenca(x, y) = |Atual(x, y) - Fundo(x, y)|
        Isola alterações reais no ambiente causadas por movimentação humana.
        """
        altura = len(matriz_atual)
        largura = len(matriz_atual[0])

        matriz_diferenca = []
        for y in range(altura):
            linha = []
            for x in range(largura):
                diff = abs(matriz_atual[y][x] - matriz_fundo_referencia[y][x])
                linha.append(diff)
            matriz_diferenca.append(linha)

        return matriz_diferenca

    @staticmethod
    def aplicar_limiarizacao(matriz_cinza, limiar=28):
        """
        Binarização manual por Thresholding:
        Se pixel >= limiar -> 255 (Pixel ativo / presença detectada)
        Se pixel < limiar  -> 0   (Fundo estático)
        """
        altura = len(matriz_cinza)
        largura = len(matriz_cinza[0])

        matriz_binaria = []
        for y in range(altura):
            linha = []
            for x in range(largura):
                pixel = matriz_cinza[y][x]
                linha.append(255 if pixel >= limiar else 0)
            matriz_binaria.append(linha)

        return matriz_binaria

    @classmethod
    def filtro_morfologico_abertura(cls, matriz_binaria):
        """
        Morfologia Matemática manual: Abertura (Erosão 3x3 seguida de Dilatação 3x3).
        Remove pequenas flutuações de iluminação e ruídos isolados mantendo
        a massa corporal do ocupante.
        """
        altura = len(matriz_binaria)
        largura = len(matriz_binaria[0])

        # 1. Erosão 3x3
        matriz_erodida = [[0 for _ in range(largura)] for _ in range(altura)]
        for y in range(1, altura - 1):
            for x in range(1, largura - 1):
                eh_objeto = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if matriz_binaria[y + dy][x + dx] != 255:
                            eh_objeto = False
                            break
                    if not eh_objeto:
                        break
                matriz_erodida[y][x] = 255 if eh_objeto else 0

        # 2. Dilatação 3x3
        matriz_dilatada = [[0 for _ in range(largura)] for _ in range(altura)]
        for y in range(1, altura - 1):
            for x in range(1, largura - 1):
                if matriz_erodida[y][x] == 255:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            matriz_dilatada[y + dy][x + dx] = 255

        return matriz_dilatada

    @classmethod
    def processar_pipeline_completo(cls, caminho_imagem, diretorio_saida=None, caminho_fundo=None):
        """
        Executa as 6 etapas completas do pipeline de Processamento Digital de Imagens (PDI):
        1. Conversão para tons de cinza (Norma ITU-R BT.601)
        2. Suavização Gaussiana passa-baixas 3x3
        3. Realce de bordas direcionais via Sobel (Gx e Gy)
        4. Subtração de imagem de plano de fundo
        5. Binarização rígida por limiar
        6. Abertura morfológica (erosão 3x3 seguida de dilatação 3x3)

        :param caminho_imagem: Caminho do arquivo da cena atual ou objeto PIL.Image.Image.
        :param diretorio_saida: Diretório opcional para salvar as 6 evidências em formato PNG.
        :param caminho_fundo: Caminho opcional do arquivo de calibração de fundo estático.
        :return: Dicionário contendo as imagens/matrizes das 6 etapas.
        """
        if isinstance(caminho_imagem, str):
            imagem_atual = cls.carregar_imagem(caminho_imagem)
        else:
            imagem_atual = caminho_imagem.convert("RGB")

        if caminho_fundo is not None:
            if isinstance(caminho_fundo, str):
                imagem_fundo = cls.carregar_imagem(caminho_fundo)
            else:
                imagem_fundo = caminho_fundo.convert("RGB")
        else:
            padrao_fundo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens_reais", "escritorio_vazio.jpg")
            if os.path.exists(padrao_fundo):
                imagem_fundo = cls.carregar_imagem(padrao_fundo)
            else:
                imagem_fundo = imagem_atual

        # 1. Tons de Cinza
        matriz_cinza = cls.converter_para_tons_de_cinza(imagem_atual)
        matriz_fundo_cinza = cls.converter_para_tons_de_cinza(imagem_fundo)

        # 2. Suavização Gaussiana 3x3
        matriz_gaussiano = cls.filtro_suavizacao_gaussiana_3x3(matriz_cinza)
        matriz_fundo_gaussiano = cls.filtro_suavizacao_gaussiana_3x3(matriz_fundo_cinza)

        # 3. Realce Sobel
        matriz_sobel = cls.filtro_realce_bordas_sobel(matriz_cinza)

        # 4. Subtração de Fundo
        matriz_subtracao = cls.subtrair_imagens(matriz_gaussiano, matriz_fundo_gaussiano)

        # 5. Binarização
        matriz_binarizada = cls.aplicar_limiarizacao(matriz_subtracao, limiar=28)

        # 6. Abertura Morfológica
        matriz_morfologia = cls.filtro_morfologico_abertura(matriz_binarizada)

        matrizes = {
            "tons_de_cinza": matriz_cinza,
            "gaussiano": matriz_gaussiano,
            "sobel": matriz_sobel,
            "subtracao_fundo": matriz_subtracao,
            "binarizada": matriz_binarizada,
            "morfologia_final": matriz_morfologia,
        }

        if diretorio_saida is not None:
            os.makedirs(diretorio_saida, exist_ok=True)
            cls.salvar_imagem(matriz_cinza, os.path.join(diretorio_saida, "1_tons_de_cinza.png"))
            cls.salvar_imagem(matriz_gaussiano, os.path.join(diretorio_saida, "2_suavizacao_gaussiana.png"))
            cls.salvar_imagem(matriz_sobel, os.path.join(diretorio_saida, "3_realce_sobel.png"))
            cls.salvar_imagem(matriz_subtracao, os.path.join(diretorio_saida, "4_subtracao_fundo.png"))
            cls.salvar_imagem(matriz_binarizada, os.path.join(diretorio_saida, "5_binarizacao_limiar.png"))
            cls.salvar_imagem(matriz_morfologia, os.path.join(diretorio_saida, "6_morfologia_final.png"))
            cls.gerar_painel_comparativo(
                imagem_atual,
                matriz_cinza,
                matriz_gaussiano,
                matriz_sobel,
                matriz_subtracao,
                matriz_morfologia,
                caminho_saida=os.path.join(diretorio_saida, "painel_comparativo_pdi.png")
            )

        altura = len(matriz_cinza)
        largura = len(matriz_cinza[0])
        imagens = {}
        for nome, m in matrizes.items():
            img = Image.new("L", (largura, altura))
            pixels = [int(max(0, min(255, round(m[y][x])))) for y in range(altura) for x in range(largura)]
            img.putdata(pixels)
            img.matriz = m
            imagens[nome] = img

        return imagens

    @classmethod
    def gerar_painel_comparativo(cls, imagem_original, matriz_cinza, matriz_suave, matriz_sobel,
                                 matriz_subtracao, matriz_morfologia, caminho_saida):
        """
        Gera um mosaico analítico 2x3 reunindo a cena original e as 5 etapas de PDI com legendas formais.
        Utilizado como elemento visual sintético para a monografia acadêmica e apresentação de slides.
        """
        from PIL import ImageDraw, ImageFont

        altura = len(matriz_cinza)
        largura = len(matriz_cinza[0])

        def matriz_para_pil(m, modo="L"):
            img = Image.new(modo, (largura, altura))
            pix = [int(max(0, min(255, round(m[y][x])))) for y in range(altura) for x in range(largura)]
            img.putdata(pix)
            return img

        if isinstance(imagem_original, Image.Image):
            img_orig = imagem_original.convert("RGB").resize((largura, altura), Image.Resampling.BILINEAR)
        else:
            img_orig = cls.carregar_imagem(imagem_original, redimensionar=(largura, altura))

        img_cinza = matriz_para_pil(matriz_cinza).convert("RGB")
        img_suave = matriz_para_pil(matriz_suave).convert("RGB")
        img_sobel = matriz_para_pil(matriz_sobel).convert("RGB")
        img_sub = matriz_para_pil(matriz_subtracao).convert("RGB")
        img_morf = matriz_para_pil(matriz_morfologia).convert("RGB")

        paineis = [
            ("1. Cena Original (RGB)", img_orig),
            ("2. Tons de Cinza (BT.601)", img_cinza),
            ("3. Suavizacao Gaussiana 3x3", img_suave),
            ("4. Realce de Bordas Sobel 3x3", img_sobel),
            ("5. Subtracao de Fundo (|I - B|)", img_sub),
            ("6. Morfologia Final (Abertura)", img_morf),
        ]

        margem_x, margem_y = 15, 15
        cabecalho_alt = 40
        rotulo_alt = 24
        largura_total = 3 * largura + 4 * margem_x
        altura_total = cabecalho_alt + 2 * (altura + rotulo_alt) + 3 * margem_y

        mosaico = Image.new("RGB", (largura_total, altura_total), (245, 247, 250))
        draw = ImageDraw.Draw(mosaico)
        font = ImageFont.load_default()

        draw.rectangle([(0, 0), (largura_total, cabecalho_alt)], fill=(30, 41, 59))
        draw.text((margem_x, 14), "SISTEMA VIGIA - ETAPAS DO PROCESSAMENTO DIGITAL DE IMAGENS (APS UNIP CC)", fill=(255, 255, 255), font=font)

        for idx, (titulo, sub_img) in enumerate(paineis):
            linha = idx // 3
            coluna = idx % 3
            pos_x = margem_x + coluna * (largura + margem_x)
            pos_y = cabecalho_alt + margem_y + linha * (altura + rotulo_alt + margem_y)

            draw.rectangle([(pos_x, pos_y), (pos_x + largura, pos_y + rotulo_alt)], fill=(51, 65, 85))
            draw.text((pos_x + 8, pos_y + 6), titulo, fill=(255, 255, 255), font=font)
            mosaico.paste(sub_img, (pos_x, pos_y + rotulo_alt))

        os.makedirs(os.path.dirname(os.path.abspath(caminho_saida)), exist_ok=True)
        mosaico.save(caminho_saida)
        return caminho_saida


# ===================================================================================
# 2. MÓDULO DE VISÃO COMPUTACIONAL (CLASSIFICADOR DE PRESENÇA)
# ===================================================================================

class DetectorPresencaVigIA:
    """
    Classificador de visão computacional e inferência neural convolucional (CNN)
    para detecção e confirmação de presença humana em ambientes prediais.
    Combina processamento digital de imagens artesanal (segmentação morfológica e
    densidade espacial de ocupação) com inferência neural profunda supervisionada
    via rede convolucional Keras/TensorFlow.
    """

    def __init__(self, caminho_modelo="modelo_vigia.keras", limiar_sensibilidade=25, percentual_minimo_ocupacao=0.8):
        """
        :param caminho_modelo: Caminho do arquivo do modelo serializado Keras (.keras).
        :param limiar_sensibilidade: Tolerância fotométrica para subtração de plano de fundo.
        :param percentual_minimo_ocupacao: Limiar percentual mínimo da área útil ocupada.
        """
        self.caminho_modelo = caminho_modelo
        self.limiar_sensibilidade = limiar_sensibilidade
        self.percentual_minimo_ocupacao = percentual_minimo_ocupacao
        self.modelo = None
        self._carregar_modelo()

    def _carregar_modelo(self):
        """
        Carrega o modelo convolucional pré-treinado salvo em disco.
        """
        candidatos = [
            self.caminho_modelo,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.caminho_modelo) if self.caminho_modelo else None
        ]
        for c in candidatos:
            if c and os.path.exists(c):
                try:
                    import keras
                    self.modelo = keras.models.load_model(c)
                    break
                except Exception:
                    pass

    def classificar_neural(self, imagem_pil_ou_caminho):
        """
        Executa a inferência neural da mini-CNN Keras.
        Retorna a probabilidade estimada de presença humana no intervalo [0.0, 1.0].
        """
        if self.modelo is None:
            self._carregar_modelo()
        if self.modelo is None:
            return None

        try:
            if isinstance(imagem_pil_ou_caminho, str):
                img = ProcessadorImagem.carregar_imagem(imagem_pil_ou_caminho, redimensionar=None)
            elif isinstance(imagem_pil_ou_caminho, Image.Image):
                img = imagem_pil_ou_caminho.convert("RGB")
            elif hasattr(imagem_pil_ou_caminho, "shape"):
                arr = imagem_pil_ou_caminho
                if len(arr.shape) == 3:
                    arr = np.expand_dims(arr, axis=0)
                if arr.shape[1:3] != (64, 64):
                    pil_tmp = Image.fromarray(arr[0].astype(np.uint8)).resize((64, 64))
                    arr = np.expand_dims(np.array(pil_tmp, dtype=np.float32), axis=0)
                prob = float(self.modelo(arr, training=False)[0][0])
                return prob
            else:
                return None

            img_redim = img.resize((64, 64), Image.Resampling.BILINEAR)
            tensor = np.expand_dims(np.array(img_redim, dtype=np.float32), axis=0)
            prob = float(self.modelo(tensor, training=False)[0][0])
            return prob
        except Exception:
            return None

    def analisar_quadro(self, matriz_atual_suave, matriz_fundo_suave, imagem_original=None):
        """
        Executa a segmentação e calcula os índices estatísticos de presença na imagem,
        integrando a inferência neural da mini-CNN quando disponível.
        """
        altura = len(matriz_atual_suave)
        largura = len(matriz_atual_suave[0])
        total_pixels = altura * largura

        # 1. Subtração de Fundo
        matriz_diferenca = ProcessadorImagem.subtrair_imagens(matriz_atual_suave, matriz_fundo_suave)

        # 2. Limiarização
        matriz_binaria = ProcessadorImagem.aplicar_limiarizacao(matriz_diferenca, self.limiar_sensibilidade)

        # 3. Morfologia de Abertura
        matriz_filtrada = ProcessadorImagem.filtro_morfologico_abertura(matriz_binaria)

        # 4. Contagem e densidade espacial
        pixels_ativos = 0
        soma_x = 0
        soma_y = 0

        for y in range(altura):
            for x in range(largura):
                if matriz_filtrada[y][x] == 255:
                    pixels_ativos += 1
                    soma_x += x
                    soma_y += y

        percentual_ocupado = (pixels_ativos / total_pixels) * 100.0

        centroide = None
        if pixels_ativos > 0:
            centroide = (int(soma_x / pixels_ativos), int(soma_y / pixels_ativos))

        presenca_pdi = percentual_ocupado >= self.percentual_minimo_ocupacao

        # 5. Inferência Neural supervisionada com mini-CNN Keras
        prob_neural = None
        if imagem_original is not None:
            prob_neural = self.classificar_neural(imagem_original)

        if prob_neural is not None:
            # Fusão ponderada de decisão PDI e Rede Convolucional Keras:
            # 1. Área segmentada nítida pelo PDI (>= 2.5%): confirma ocupação humana imediata
            # 2. Área no limiar (0.8% a 2.5%): validação cruzada exigindo predição da CNN (>= 0.40)
            # 3. Área insignificante (< 0.8%): sala desocupada (ruídos ópticos/eletrônicos eliminados)
            if percentual_ocupado >= 2.5:
                presenca_confirmada = True
                confianca = max(80.0, min(99.9, 70.0 + percentual_ocupado * 2.0))
            elif percentual_ocupado >= self.percentual_minimo_ocupacao:
                if prob_neural >= 0.40:
                    presenca_confirmada = True
                    confianca = max(70.0, prob_neural * 100.0)
                else:
                    presenca_confirmada = False
                    confianca = max(70.0, (1.0 - prob_neural) * 100.0)
            else:
                presenca_confirmada = False
                confianca = max(75.0, (1.0 - prob_neural) * 100.0)
        else:
            presenca_confirmada = presenca_pdi
            if presenca_confirmada:
                confianca = min(99.9, 65.0 + (percentual_ocupado * 8.0))
            else:
                confianca = max(10.0, 100.0 - (percentual_ocupado * 25.0))

        relatorio = {
            "presenca_detectada": presenca_confirmada,
            "percentual_ocupado": round(percentual_ocupado, 2),
            "pixels_ativos": pixels_ativos,
            "total_pixels": total_pixels,
            "centroide": centroide,
            "confianca_percentual": round(confianca, 1),
            "probabilidade_neural": round(prob_neural, 6) if prob_neural is not None else None,
            "matrizes_intermediarias": {
                "diferenca": matriz_diferenca,
                "binaria": matriz_binaria,
                "morfologia": matriz_filtrada
            }
        }

        return relatorio

    def analisar_presenca(self, caminho_imagem, imagem_processada=None):
        """
        Interface formal do módulo de Visão Computacional conforme contrato de projeto:
        Realiza a inferência combinada (PDI + mini-CNN).
        Retorna (presenca_confirmada: bool, confianca: float, detalhes: dict).
        Trata exceções graciosamente com retorno de presença falsa em caso de ausência de calibração ou erro de leitura.
        """
        try:
            if isinstance(caminho_imagem, str) and not os.path.exists(caminho_imagem) and imagem_processada is None:
                return False, 0.0, {"erro": "Arquivo de imagem não encontrado", "presenca_detectada": False}

            img = imagem_processada
            if img is None:
                img = ProcessadorImagem.carregar_imagem(caminho_imagem)

            prob_neural = self.classificar_neural(img)
            if prob_neural is None:
                return False, 0.0, {"erro": "Rede neural indisponível", "presenca_detectada": False}

            presenca_confirmada = prob_neural >= 0.5
            confianca = max(70.0, (prob_neural if presenca_confirmada else (1.0 - prob_neural)) * 100.0)
            detalhes = {
                "presenca_detectada": presenca_confirmada,
                "probabilidade_neural": round(prob_neural, 6),
                "confianca_percentual": round(confianca, 1),
                "caminho_imagem": caminho_imagem if isinstance(caminho_imagem, str) else None
            }
            return presenca_confirmada, round(confianca, 1), detalhes
        except Exception as e:
            return False, 0.0, {"erro": str(e), "presenca_detectada": False}


# ===================================================================================
# 3. SISTEMA INTEGRADO VIGIA (CONTROLE, SENSORES E SUSTENTABILIDADE)
# ===================================================================================

class SistemaVigIA:
    """
    Controlador central do sistema VigIA conforme proposto nos slides:
    - Gerencia o estado das lâmpadas (LIGADA / DESLIGADA).
    - Simula o sensor PIR de infravermelho de baixo custo (sentinela).
    - Aciona a câmera sob demanda, economizando processamento.
    - Calcula o impacto ambiental e a economia de energia em tempo real.
    """

    def __init__(self, potencia_iluminacao_watts=400, tarifa_kwh=0.85):
        self.potencia_watts = potencia_iluminacao_watts
        self.tarifa_kwh = tarifa_kwh

        self.luz_acesa = True
        self.modo_sentinela_pir = False

        self.horas_totais_monitoradas = 0.0
        self.horas_luz_economizada = 0.0

        self.processador = ProcessadorImagem()
        self.detector = DetectorPresencaVigIA()
        self.matriz_fundo_referencia = None

    def calibrar_fundo(self, caminho_imagem_sala_vazia):
        """
        Calibração com a foto real ou quadro de vídeo do ambiente vazio.
        """
        if isinstance(caminho_imagem_sala_vazia, str):
            nome = os.path.basename(caminho_imagem_sala_vazia)
        else:
            nome = "Quadro_Calibracao_Webcam"
        print(f"\n[CALIBRAÇÃO VIGIA] Carregando imagem real de fundo: {nome}")
        imagem = self.processador.carregar_imagem(caminho_imagem_sala_vazia)
        matriz_cinza = self.processador.converter_para_tons_de_cinza(imagem)
        self.matriz_fundo_referencia = self.processador.filtro_suavizacao_gaussiana_3x3(matriz_cinza)
        print(" -> Calibração concluída com sucesso (tons de cinza + filtro Gaussiano manual).")

    def processar_ciclo(self, caminho_imagem_atual, sinal_sensor_pir=False, duracao_minutos=15, pasta_saida_debug=None):
        """
        Ciclo de tomada de decisão do VigIA:
        1. Se a luz está desligada e não há PIR, mantém lâmpadas e câmera desligadas.
        2. Se há PIR ou luz acesa, processa imagem e decide se apaga, mantém ou religa.
        """
        self.horas_totais_monitoradas += (duracao_minutos / 60.0)

        print("\n" + "="*70)
        if isinstance(caminho_imagem_atual, str):
            nome_arquivo = os.path.basename(caminho_imagem_atual)
        else:
            nome_arquivo = "Quadro_Webcam_AoVivo"
        print(f"CICLO DE MONITORAMENTO VIGIA | Imagem: {nome_arquivo} | Intervalo: {duracao_minutos} min")
        print(f"Estado Inicial: LUZ={'LIGADA' if self.luz_acesa else 'DESLIGADA'} | Sensor PIR={'ATIVO' if sinal_sensor_pir else 'INATIVO'}")
        print("="*70)

        if self.matriz_fundo_referencia is None:
            raise RuntimeError("O sistema não foi calibrado! Chame calibrar_fundo() primeiro.")

        deve_processar = self.luz_acesa or sinal_sensor_pir

        if not deve_processar:
            print("[MODO SENTINELA] Ninguém no ambiente e sem calor/movimento PIR. Lâmpadas apagadas.")
            self.horas_luz_economizada += (duracao_minutos / 60.0)
            return {
                "luz_ligada": False,
                "presenca": False,
                "motivo": "Sentinela PIR em guarda",
                "ocupacao_percentual": 0.0,
                "confianca": 100.0,
                "probabilidade_neural": 0.0
            }

        # 1. Aquisição da imagem real atual
        imagem_atual = self.processador.carregar_imagem(caminho_imagem_atual)
        matriz_cinza = self.processador.converter_para_tons_de_cinza(imagem_atual)

        # 2. Suavização Gaussiana manual
        matriz_suave = self.processador.filtro_suavizacao_gaussiana_3x3(matriz_cinza)

        # 3. Realce de Bordas Sobel manual
        matriz_bordas_sobel = self.processador.filtro_realce_bordas_sobel(matriz_cinza)

        # 4. Detecção por Visão Computacional e Inferência Neural
        resultado = self.detector.analisar_quadro(
            matriz_suave, self.matriz_fundo_referencia, imagem_original=imagem_atual
        )
        presenca = resultado["presenca_detectada"]

        print(f"[PDI & VISÃO] Pixels ativos: {resultado['pixels_ativos']} ({resultado['percentual_ocupado']}%)")
        prob_str = f" | Probabilidade CNN: {resultado['probabilidade_neural']:.4f}" if resultado.get("probabilidade_neural") is not None else ""
        print(f"[INFERÊNCIA NEURAL] Presença Humana: {'SIM' if presenca else 'NÃO'}{prob_str} | Confiança: {resultado['confianca_percentual']}%")

        # 5. Lógica de Iluminação Sustentável
        if presenca:
            if not self.luz_acesa:
                print(" -> [AÇÃO VIGIA] Movimento confirmado! RELIGANDO AS LUZES IMEDIATAMENTE.")
            else:
                print(" -> [AÇÃO VIGIA] Sala ocupada! MANTENDO AS LUZES ACESAS.")
            self.luz_acesa = True
            self.modo_sentinela_pir = False
        else:
            if self.luz_acesa:
                print(" -> [AÇÃO VIGIA] Sala desocupada! APAGANDO LUZES para economizar energia.")
            else:
                print(" -> [AÇÃO VIGIA] Sala permanece vazia. Luzes continuam desligadas.")
            self.luz_acesa = False
            self.modo_sentinela_pir = True
            self.horas_luz_economizada += (duracao_minutos / 60.0)

        # 6. Salvar evidências visuais das 6 etapas para a monografia da APS
        if pasta_saida_debug:
            os.makedirs(pasta_saida_debug, exist_ok=True)
            self.processador.salvar_imagem(matriz_cinza, os.path.join(pasta_saida_debug, "1_tons_de_cinza.png"))
            self.processador.salvar_imagem(matriz_suave, os.path.join(pasta_saida_debug, "2_suavizacao_gaussiana.png"))
            self.processador.salvar_imagem(matriz_bordas_sobel, os.path.join(pasta_saida_debug, "3_realce_sobel.png"))
            self.processador.salvar_imagem(resultado["matrizes_intermediarias"]["diferenca"], os.path.join(pasta_saida_debug, "4_subtracao_fundo.png"))
            self.processador.salvar_imagem(resultado["matrizes_intermediarias"]["binaria"], os.path.join(pasta_saida_debug, "5_binarizacao_limiar.png"))
            self.processador.salvar_imagem(resultado["matrizes_intermediarias"]["morfologia"], os.path.join(pasta_saida_debug, "6_morfologia_final.png"))
            self.processador.gerar_painel_comparativo(
                imagem_atual,
                matriz_cinza,
                matriz_suave,
                matriz_bordas_sobel,
                resultado["matrizes_intermediarias"]["diferenca"],
                resultado["matrizes_intermediarias"]["morfologia"],
                caminho_saida=os.path.join(pasta_saida_debug, "painel_comparativo_pdi.png")
            )
            print(f"[EVIDÊNCIAS APS] 6 etapas e painel comparativo salvos em: '{pasta_saida_debug}'")

        return {
            "luz_ligada": self.luz_acesa,
            "presenca": presenca,
            "ocupacao_percentual": resultado["percentual_ocupado"],
            "confianca": resultado["confianca_percentual"],
            "probabilidade_neural": resultado.get("probabilidade_neural")
        }

    def obter_relatorio_sustentabilidade(self):
        """
        Retorna as métricas de sustentabilidade e eficiência energética em formato numérico estruturado.

        Fórmulas matemáticas:
        - kwh_convencional = (potencia_watts * horas_totais) / 1000.0
        - kwh_vigia = (potencia_watts * horas_luz_acesa) / 1000.0
        - kwh_poupados = kwh_convencional - kwh_vigia
        - percentual_economia = (kwh_poupados / kwh_convencional * 100.0) se kwh_convencional > 0 senão 0.0
        - economia_reais = kwh_poupados * tarifa_kwh
        - reducao_co2_kg = kwh_poupados * 0.084 (fator médio anual do SIN / MCTI)
        """
        kwh_convencional = (self.potencia_watts * self.horas_totais_monitoradas) / 1000.0
        horas_luz_ligada = max(0.0, self.horas_totais_monitoradas - self.horas_luz_economizada)
        kwh_vigia = (self.potencia_watts * horas_luz_ligada) / 1000.0

        kwh_poupados = kwh_convencional - kwh_vigia
        percentual_economia = (kwh_poupados / kwh_convencional * 100.0) if kwh_convencional > 0 else 0.0
        economia_reais = kwh_poupados * self.tarifa_kwh
        reducao_co2_kg = kwh_poupados * 0.084

        return {
            "kwh_convencional": round(float(kwh_convencional), 4),
            "kwh_vigia": round(float(kwh_vigia), 4),
            "kwh_poupados": round(float(kwh_poupados), 4),
            "economia_reais": round(float(economia_reais), 4),
            "reducao_co2_kg": round(float(reducao_co2_kg), 4),
            "percentual_economia": round(float(percentual_economia), 4)
        }

    def gerar_relatorio_sustentabilidade(self):
        """
        Retorna a síntese textual formatada do balanço energético e ambiental do VigIA.
        """
        dados = self.obter_relatorio_sustentabilidade()
        relatorio = f"""
===================================================================================
           RELATÓRIO DE SUSTENTABILIDADE E EFICIÊNCIA ENERGÉTICA (VigIA)
===================================================================================
Tempo Total Monitorado           : {self.horas_totais_monitoradas:.1f} horas
Tempo de Luz Poupada (Apagada)   : {self.horas_luz_economizada:.1f} horas ({dados['percentual_economia']:.1f}% de tempo desligado)
Potência Instalada da Sala       : {self.potencia_watts} Watts
-----------------------------------------------------------------------------------
Consumo Sem Automação            : {dados['kwh_convencional']:.2f} kWh
Consumo Inteligente VigIA        : {dados['kwh_vigia']:.2f} kWh
ENERGIA TOTAL POUPADA            : {dados['kwh_poupados']:.2f} kWh ({dados['percentual_economia']:.1f}% de redução)
ECONOMIA FINANCEIRA ESTIMADA     : R$ {dados['economia_reais']:.2f}
REDUÇÃO DE PEGADA DE CARBONO     : {dados['reducao_co2_kg']:.3f} kg de CO2 evitados
===================================================================================
"""
        return relatorio


# ===================================================================================
# 4. EXECUÇÃO PRINCIPAL COM IMAGENS REAIS
# ===================================================================================

def executar_demonstracao_vigia():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    pasta_imagens_reais = os.path.join(diretorio_atual, "imagens_reais")
    pasta_etapas = os.path.join(diretorio_atual, "etapas_processamento")

    # Arquivos de imagens reais de escritório
    img_vazio = os.path.join(pasta_imagens_reais, "escritorio_vazio.jpg")
    img_ocupado = os.path.join(pasta_imagens_reais, "escritorio_com_pessoas.jpg")

    print("\n" + "#"*70)
    print("   INICIALIZANDO VIGIA COM IMAGENS REAIS DE ESCRITÓRIO E PESSOAS")
    print("#"*70)

    # 1. Instancia o sistema com carga de 400W (sala padrão de faculdade/escritório)
    sistema = SistemaVigIA(potencia_iluminacao_watts=400, tarifa_kwh=0.85)

    # 2. Calibração inicial com a imagem real do escritório vazio
    sistema.calibrar_fundo(img_vazio)

    # --- CICLO 1: Pessoas entram e trabalham no escritório ---
    # Situação: Horário comercial/estudos, pessoas no escritório.
    # O VigIA detecta ocupação em massa e mantém a iluminação acesa.
    sistema.processar_ciclo(
        caminho_imagem_atual=img_ocupado,
        sinal_sensor_pir=True,
        duracao_minutos=60,
        pasta_saida_debug=pasta_etapas
    )

    # --- CICLO 2: Fim de expediente (Pessoas saem do escritório) ---
    # Situação: Escritório esvaziou. O VigIA detecta ausência e APAGA AS LUZES!
    sistema.processar_ciclo(
        caminho_imagem_atual=img_vazio,
        sinal_sensor_pir=False,
        duracao_minutos=30
    )

    # --- CICLO 3: Madrugada / Noite (Escritório vazio e escuro) ---
    # Situação: Sensor PIR em guarda (Modo Sentinela). Câmera em repouso.
    sistema.processar_ciclo(
        caminho_imagem_atual=img_vazio,
        sinal_sensor_pir=False,
        duracao_minutos=180
    )

    # --- CICLO 4: Alguém retorna à noite (Gatilho PIR acionado) ---
    # Situação: Funcionário/aluno entra para pegar documento.
    # O PIR detecta calor/movimento -> Acorda câmera -> Confirma presença -> Religa luz!
    sistema.processar_ciclo(
        caminho_imagem_atual=img_ocupado,
        sinal_sensor_pir=True,
        duracao_minutos=15
    )

    # 3. Exibe o relatório de sustentabilidade
    print(sistema.gerar_relatorio_sustentabilidade())


def executar_modo_webcam():
    """
    Inicia a interface gráfica com câmera ao vivo e máscara de PDI em tempo real.
    Utiliza Pygame nativo (zero OpenCV) para exibição fluida da câmera e indicadores.
    """
    try:
        from interface_webcam import executar_interface_webcam
        executar_interface_webcam()
    except Exception as e:
        print(f"\n[ERRO AO INICIALIZAR INTERFACE] {e}")
        print("Certifique-se de que o ambiente gráfico e as dependências estão ativos:")
        print("  pip install pygame pillow numpy tensorflow")


if __name__ == "__main__":
    if "--webcam" in sys.argv:
        executar_modo_webcam()
    else:
        executar_demonstracao_vigia()
