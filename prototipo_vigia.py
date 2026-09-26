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
        Carrega uma imagem real do disco e converte para RGB.
        Aplica redimensionamento proporcional para resolução padrão de processamento
        (padrão da indústria de CFTV para garantir velocidade em tempo real).
        """
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


# ===================================================================================
# 2. MÓDULO DE VISÃO COMPUTACIONAL (CLASSIFICADOR DE PRESENÇA)
# ===================================================================================

class DetectorPresencaVigIA:
    """
    Modelo de Visão Computacional para detecção de ocupação e presença humana.
    Analisa a densidade espacial, contagem de pixels ativos e geometria.
    """

    def __init__(self, limiar_sensibilidade=25, percentual_minimo_ocupacao=0.8):
        """
        :param limiar_sensibilidade: Tolerância de luminosidade para subtrair fundo.
        :param percentual_minimo_ocupacao: % mínimo da cena ocupado por corpo humano.
        """
        self.limiar_sensibilidade = limiar_sensibilidade
        self.percentual_minimo_ocupacao = percentual_minimo_ocupacao

    def analisar_quadro(self, matriz_atual_suave, matriz_fundo_suave):
        """
        Executa a segmentação e calcula os índices estatísticos de presença na imagem.
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

        presenca_confirmada = percentual_ocupado >= self.percentual_minimo_ocupacao

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
            "matrizes_intermediarias": {
                "diferenca": matriz_diferenca,
                "binaria": matriz_binaria,
                "morfologia": matriz_filtrada
            }
        }

        return relatorio


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
        Calibração com a foto real do ambiente vazio.
        """
        print(f"\n[CALIBRAÇÃO VIGIA] Carregando imagem real de fundo: {os.path.basename(caminho_imagem_sala_vazia)}")
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
        nome_arquivo = os.path.basename(caminho_imagem_atual)
        print(f"CICLO DE MONITORAMENTO VIGIA | Imagem: {nome_arquivo} | Intervalo: {duracao_minutos} min")
        print(f"Estado Inicial: LUZ={'LIGADA' if self.luz_acesa else 'DESLIGADA'} | Sensor PIR={'ATIVO' if sinal_sensor_pir else 'INATIVO'}")
        print("="*70)

        if self.matriz_fundo_referencia is None:
            raise RuntimeError("O sistema não foi calibrado! Chame calibrar_fundo() primeiro.")

        deve_processar = self.luz_acesa or sinal_sensor_pir

        if not deve_processar:
            print("[MODO SENTINELA] Ninguém no ambiente e sem calor/movimento PIR. Lâmpadas apagadas.")
            self.horas_luz_economizada += (duracao_minutos / 60.0)
            return {"luz_ligada": False, "motivo": "Sentinela PIR em guarda"}

        # 1. Aquisição da imagem real atual
        imagem_atual = self.processador.carregar_imagem(caminho_imagem_atual)
        matriz_cinza = self.processador.converter_para_tons_de_cinza(imagem_atual)

        # 2. Suavização Gaussiana manual
        matriz_suave = self.processador.filtro_suavizacao_gaussiana_3x3(matriz_cinza)

        # 3. Realce de Bordas Sobel manual
        matriz_bordas_sobel = self.processador.filtro_realce_bordas_sobel(matriz_cinza)

        # 4. Detecção por Visão Computacional
        resultado = self.detector.analisar_quadro(matriz_suave, self.matriz_fundo_referencia)
        presenca = resultado["presenca_detectada"]

        print(f"[PDI & VISÃO] Pixels ativos: {resultado['pixels_ativos']} ({resultado['percentual_ocupado']}%)")
        print(f"[DECISÃO IA] Presença Humana: {'SIM' if presenca else 'NÃO'} | Confiança: {resultado['confianca_percentual']}%")

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
            print(f"[EVIDÊNCIAS APS] 6 etapas de PDI salvas em: '{pasta_saida_debug}'")

        return {
            "luz_ligada": self.luz_acesa,
            "presenca": presenca,
            "ocupacao_percentual": resultado["percentual_ocupado"],
            "confianca": resultado["confianca_percentual"]
        }

    def gerar_relatorio_sustentabilidade(self):
        """
        Métricas ambientais e econômicas do VigIA.
        """
        kwh_convencional = (self.potencia_watts * self.horas_totais_monitoradas) / 1000.0
        horas_luz_ligada = max(0.0, self.horas_totais_monitoradas - self.horas_luz_economizada)
        kwh_vigia = (self.potencia_watts * horas_luz_ligada) / 1000.0

        kwh_poupados = kwh_convencional - kwh_vigia
        porcentagem_economia = (kwh_poupados / kwh_convencional * 100.0) if kwh_convencional > 0 else 0.0
        economia_reais = kwh_poupados * self.tarifa_kwh
        co2_evitado_kg = kwh_poupados * 0.084

        relatorio = f"""
===================================================================================
           RELATÓRIO DE SUSTENTABILIDADE E EFICIÊNCIA ENERGÉTICA (VigIA)
===================================================================================
Tempo Total Monitorado           : {self.horas_totais_monitoradas:.1f} horas
Tempo de Luz Poupada (Apagada)   : {self.horas_luz_economizada:.1f} horas ({porcentagem_economia:.1f}% de tempo desligado)
Potência Instalada da Sala       : {self.potencia_watts} Watts
-----------------------------------------------------------------------------------
Consumo Sem Automação            : {kwh_convencional:.2f} kWh
Consumo Inteligente VigIA        : {kwh_vigia:.2f} kWh
ENERGIA TOTAL POUPADA            : {kwh_poupados:.2f} kWh ({porcentagem_economia:.1f}% de redução)
ECONOMIA FINANCEIRA ESTIMADA     : R$ {economia_reais:.2f}
REDUÇÃO DE PEGADA DE CARBONO     : {co2_evitado_kg:.3f} kg de CO2 evitados
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


if __name__ == "__main__":
    executar_demonstracao_vigia()
