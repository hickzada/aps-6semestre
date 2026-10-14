"""
===================================================================================
PROJETO APS - CIÊNCIA DA COMPUTAÇÃO (5º/6º SEMESTRE) - UNIP
TEMA: Aplicação de Técnicas de Processamento Digital de Imagens e Visão 
      Computacional para Resolver Problemas de Sustentabilidade Ambiental.
SISTEMA: VigIA - Interface Gráfica Interativa com Câmera ao Vivo
===================================================================================
Interface visual desenvolvida em Pygame nativo (zero dependência de OpenCV)
para exibição em tempo real do fluxo de vídeo da câmera, da máscara morfológica
artesanal de PDI e dos indicadores de tomada de decisão (presença, lâmpada e sustentabilidade).
===================================================================================
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# IMPORTANTE: Carrega os módulos de aprendizado profundo (TensorFlow/Keras) ANTES do Pygame
# para evitar colisão de símbolos LLVM/Mesa no subsistema gráfico do Linux
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    import keras
except ImportError:
    pass

from PIL import Image

# Importação e pré-carregamento do modelo antes da inicialização gráfica
from prototipo_vigia import SistemaVigIA, ProcessadorImagem

import pygame
import pygame.camera


def matriz_para_surface(matriz_pixels, largura_alvo=440, altura_alvo=330):
    """
    Converte uma matriz bidimensional de intensidades (0 a 255) em um pygame.Surface
    redimensionado para exibição rápida na interface sem sobrecarregar a CPU.
    """
    altura = len(matriz_pixels)
    largura = len(matriz_pixels[0])
    
    flat = bytes([int(max(0, min(255, round(matriz_pixels[y][x])))) for y in range(altura) for x in range(largura)])
    pil_img = Image.frombytes("L", (largura, altura), flat).convert("RGB")
    
    if (largura, altura) != (largura_alvo, altura_alvo):
        pil_img = pil_img.resize((largura_alvo, altura_alvo), Image.Resampling.NEAREST)
        
    return pygame.image.fromstring(pil_img.tobytes(), (largura_alvo, altura_alvo), "RGB")


def executar_interface_webcam():
    """
    Inicializa a janela interativa do VigIA com transmissão da câmera ao vivo,
    exibição da máscara de segmentação de PDI e métricas de eficiência energética.
    """
    # 1. Pré-instancia o controlador VigIA e o modelo Keras ANTES do Pygame inicializar a janela
    sistema = SistemaVigIA(potencia_iluminacao_watts=400, tarifa_kwh=0.85)

    # 2. Inicialização do Pygame e subsistemas
    pygame.init()
    pygame.font.init()
    pygame.camera.init()

    cameras = pygame.camera.list_cameras()
    if not cameras:
        print("\n[ERRO DE HARDWARE] Nenhuma webcam detectada no sistema.")
        return

    largura_janela = 1000
    altura_janela = 680
    tela = pygame.display.set_mode((largura_janela, altura_janela))
    pygame.display.set_caption("VigIA - Monitoramento Inteligente de Ocupação (UNIP CC 6º Semestre)")

    relogio = pygame.time.Clock()
    fonte_titulo = pygame.font.SysFont("Arial", 20, bold=True)
    fonte_corpo = pygame.font.SysFont("Arial", 15)
    fonte_status = pygame.font.SysFont("Arial", 18, bold=True)
    fonte_pequena = pygame.font.SysFont("Arial", 12)

    # Inicialização da câmera
    dispositivo = cameras[0]
    cam = pygame.camera.Camera(dispositivo, (640, 480))
    cam.start()

    calibrado = False
    mensagem_alerta = "Posicione a câmera para a sala vazia e aperte [ESPAÇO] para Calibrar Fundo."
    cor_alerta = (234, 179, 8)  # Amarelo alerta

    # Descarte de quadros iniciais para ajuste de fotometria da câmera
    for _ in range(5):
        cam.get_image()
        time.sleep(0.02)

    # Estado da última inferência
    ultima_inferencia = {
        "presenca": False,
        "luz_ligada": False,
        "ocupacao_percentual": 0.0,
        "confianca": 100.0,
        "probabilidade_neural": 0.0,
        "matriz_morfologia": None,
        "pixels_ativos": 0
    }

    tempo_ultimo_processamento = 0.0
    intervalo_processamento = 0.25  # Processa 4 vezes por segundo para fluidez e baixo consumo de CPU
    inicio_sessao = time.time()

    executando = True
    while executando:
        tempo_atual = time.time()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                executando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key in (pygame.K_ESCAPE, pygame.K_q):
                    executando = False
                elif evento.key == pygame.K_SPACE:
                    # Captura o quadro atual como plano de fundo de calibração
                    surf_fundo = cam.get_image()
                    raw_fundo = pygame.image.tostring(surf_fundo, "RGB")
                    img_fundo = Image.frombytes("RGB", surf_fundo.get_size(), raw_fundo)
                    sistema.calibrar_fundo(img_fundo)
                    calibrado = True
                    mensagem_alerta = "Fundo calibrado com sucesso! Monitoramento ativo."
                    cor_alerta = (34, 197, 94)  # Verde sucesso

        # Captura quadro da câmera
        surf_cam = cam.get_image()

        # Execução periódica do pipeline de PDI e inferência neural
        if calibrado and (tempo_atual - tempo_ultimo_processamento >= intervalo_processamento):
            tempo_ultimo_processamento = tempo_atual
            
            # Converte para formato PIL Image esperado pelo processador
            raw_frame = pygame.image.tostring(surf_cam, "RGB")
            img_pil = Image.frombytes("RGB", surf_cam.get_size(), raw_frame)
            
            # Executa ciclo com intervalo de tempo proporcional
            delta_minutos = (intervalo_processamento / 60.0)
            res = sistema.processar_ciclo(img_pil, sinal_sensor_pir=True, duracao_minutos=delta_minutos)
            
            ultima_inferencia["presenca"] = res["presenca"]
            ultima_inferencia["luz_ligada"] = res["luz_ligada"]
            ultima_inferencia["ocupacao_percentual"] = res["ocupacao_percentual"]
            ultima_inferencia["confianca"] = res["confianca"]
            ultima_inferencia["probabilidade_neural"] = res.get("probabilidade_neural")
            
            # Obtém a matriz de morfologia para renderização visual
            if hasattr(sistema, "detector") and sistema.matriz_fundo_referencia is not None:
                img_redim = sistema.processador.carregar_imagem(img_pil)
                cinza = sistema.processador.converter_para_tons_de_cinza(img_redim)
                suave = sistema.processador.filtro_suavizacao_gaussiana_3x3(cinza)
                dif = sistema.processador.subtrair_imagens(suave, sistema.matriz_fundo_referencia)
                binar = sistema.processador.aplicar_limiarizacao(dif, sistema.detector.limiar_sensibilidade)
                morf = sistema.processador.filtro_morfologico_abertura(binar)
                ultima_inferencia["matriz_morfologia"] = morf

        # =========================================================================
        # RENDERIZAÇÃO DA INTERFACE GRÁFICA (DASHBOARD ACADÊMICO)
        # =========================================================================
        tela.fill((15, 23, 42))  # Fundo azul-ardósia escuro (#0f172a)

        # 1. Barra de Cabeçalho Superior
        pygame.draw.rect(tela, (30, 41, 59), (0, 0, largura_janela, 65))
        txt_cabecalho = fonte_titulo.render("VigIA - Sistema Inteligente de Iluminação por PDI e Visão Computacional", True, (255, 255, 255))
        txt_sub = fonte_pequena.render("UNIP - Ciência da Computação (6º Semestre) | PDI Artesanal ITU-R BT.601 + Mini-CNN Keras", True, (148, 163, 184))
        tela.blit(txt_cabecalho, (25, 12))
        tela.blit(txt_sub, (25, 38))

        # 2. Painel da Câmera ao Vivo (Esquerda)
        quadro_w, quadro_h = 460, 345
        pos_cam_x, pos_cam_y = 25, 85
        
        # Moldura com status da lâmpada
        cor_borda = (34, 197, 94) if ultima_inferencia["luz_ligada"] else (239, 68, 68)
        pygame.draw.rect(tela, (30, 41, 59), (pos_cam_x - 4, pos_cam_y - 25, quadro_w + 8, quadro_h + 33), border_radius=6)
        pygame.draw.rect(tela, cor_borda, (pos_cam_x - 2, pos_cam_y - 2, quadro_w + 4, quadro_h + 4), width=3, border_radius=4)
        
        lbl_cam = fonte_corpo.render("CÂMERA AO VIVO (CFTV AMBIENTAL)", True, (226, 232, 240))
        tela.blit(lbl_cam, (pos_cam_x + 5, pos_cam_y - 22))
        
        surf_cam_redim = pygame.transform.scale(surf_cam, (quadro_w, quadro_h))
        tela.blit(surf_cam_redim, (pos_cam_x, pos_cam_y))

        # 3. Painel da Máscara de PDI (Direita)
        pos_pdi_x, pos_pdi_y = 515, 85
        pygame.draw.rect(tela, (30, 41, 59), (pos_pdi_x - 4, pos_pdi_y - 25, quadro_w + 8, quadro_h + 33), border_radius=6)
        pygame.draw.rect(tela, (71, 85, 105), (pos_pdi_x - 2, pos_pdi_y - 2, quadro_w + 4, quadro_h + 4), width=2, border_radius=4)
        
        lbl_pdi = fonte_corpo.render("MÁSCARA PDI ARTESANAL (ABERTURA MORFOLÓGICA)", True, (226, 232, 240))
        tela.blit(lbl_pdi, (pos_pdi_x + 5, pos_pdi_y - 22))

        if ultima_inferencia["matriz_morfologia"] is not None:
            surf_pdi = matriz_para_surface(ultima_inferencia["matriz_morfologia"], quadro_w, quadro_h)
            tela.blit(surf_pdi, (pos_pdi_x, pos_pdi_y))
        else:
            pygame.draw.rect(tela, (15, 23, 42), (pos_pdi_x, pos_pdi_y, quadro_w, quadro_h))
            txt_aviso = fonte_corpo.render("Aguardando calibração de fundo...", True, (148, 163, 184))
            tela.blit(txt_aviso, (pos_pdi_x + 90, pos_pdi_y + 160))

        # 4. Barra de Alerta e Instruções
        pygame.draw.rect(tela, (30, 41, 59), (25, 450, 950, 40), border_radius=6)
        txt_msg = fonte_corpo.render(f"Instrução: {mensagem_alerta}", True, cor_alerta)
        tela.blit(txt_msg, (40, 460))

        # 5. Painel Inferior de Métricas e Tomada de Decisão
        pos_inf_y = 505
        
        # Caixa de Status da Iluminação
        box_status_w = 300
        cor_status_bg = (20, 83, 45) if ultima_inferencia["luz_ligada"] else (127, 29, 29)
        pygame.draw.rect(tela, cor_status_bg, (25, pos_inf_y, box_status_w, 140), border_radius=8)
        pygame.draw.rect(tela, cor_borda, (25, pos_inf_y, box_status_w, 140), width=2, border_radius=8)
        
        str_status = "LÂMPADA: ACESA" if ultima_inferencia["luz_ligada"] else "LÂMPADA: APAGADA"
        str_humano = "PRESENÇA HUMANA: SIM" if ultima_inferencia["presenca"] else "PRESENÇA HUMANA: NÃO"
        
        lbl_status1 = fonte_status.render(str_status, True, (255, 255, 255))
        lbl_status2 = fonte_corpo.render(str_humano, True, (241, 245, 249))
        lbl_status3 = fonte_pequena.render(f"Confiança Geral: {ultima_inferencia['confianca']:.1f}%", True, (203, 213, 225))
        
        tela.blit(lbl_status1, (45, pos_inf_y + 25))
        tela.blit(lbl_status2, (45, pos_inf_y + 60))
        tela.blit(lbl_status3, (45, pos_inf_y + 95))

        # Cartão de Análise de PDI e CNN
        pygame.draw.rect(tela, (30, 41, 59), (345, pos_inf_y, 300, 140), border_radius=8)
        lbl_metricas_tit = fonte_status.render("Diagnóstico de Visão", True, (255, 255, 255))
        lbl_m1 = fonte_corpo.render(f"Área Segmentada (PDI): {ultima_inferencia['ocupacao_percentual']:.2f}%", True, (226, 232, 240))
        prob_cnn = ultima_inferencia.get("probabilidade_neural")
        str_cnn = f"{prob_cnn * 100.0:.1f}%" if prob_cnn is not None else "N/A"
        lbl_m2 = fonte_corpo.render(f"Probabilidade Mini-CNN: {str_cnn}", True, (226, 232, 240))
        lbl_m3 = fonte_pequena.render("Gatilho Sentinela: PIR ATIVO", True, (148, 163, 184))
        
        tela.blit(lbl_metricas_tit, (365, pos_inf_y + 20))
        tela.blit(lbl_m1, (365, pos_inf_y + 55))
        tela.blit(lbl_m2, (365, pos_inf_y + 80))
        tela.blit(lbl_m3, (365, pos_inf_y + 105))

        # Cartão de Sustentabilidade Ambiental
        pygame.draw.rect(tela, (30, 41, 59), (665, pos_inf_y, 310, 140), border_radius=8)
        dados_sust = sistema.obter_relatorio_sustentabilidade()
        
        lbl_sust_tit = fonte_status.render("Impacto Sustentável", True, (255, 255, 255))
        lbl_s1 = fonte_corpo.render(f"Energia Poupada: {dados_sust['kwh_poupados']:.3f} kWh ({dados_sust['percentual_economia']:.1f}%)", True, (74, 222, 128))
        lbl_s2 = fonte_corpo.render(f"Economia Estimada: R$ {dados_sust['economia_reais']:.2f}", True, (226, 232, 240))
        lbl_s3 = fonte_corpo.render(f"CO2 Evitado: {dados_sust['reducao_co2_kg'] * 1000.0:.1f} g de carbono", True, (56, 189, 248))
        
        tela.blit(lbl_sust_tit, (685, pos_inf_y + 20))
        tela.blit(lbl_s1, (685, pos_inf_y + 55))
        tela.blit(lbl_s2, (685, pos_inf_y + 80))
        tela.blit(lbl_s3, (685, pos_inf_y + 105))

        # Rodapé de Atalhos
        txt_atalhos = fonte_pequena.render("[ESPAÇO] Recalibrar Fundo  |  [Q] ou [ESC] Encerrar Sessão", True, (100, 116, 139))
        tela.blit(txt_atalhos, (25, 655))

        pygame.display.flip()
        relogio.tick(30)

    # Encerramento limpo
    cam.stop()
    pygame.quit()
    print("\n" + "="*70)
    print("SESSÃO DA WEBCAM FINALIZADA - RELATÓRIO CONSOLIDADO:")
    print("="*70)
    print(sistema.gerar_relatorio_sustentabilidade())


if __name__ == "__main__":
    executar_interface_webcam()
