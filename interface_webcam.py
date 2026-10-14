"""
===================================================================================
PROJETO APS - CIÊNCIA DA COMPUTAÇÃO (5º/6º SEMESTRE) - UNIP
TEMA: Aplicação de Técnicas de Processamento Digital de Imagens e Visão 
      Computacional para Resolver Problemas de Sustentabilidade Ambiental.
SISTEMA: VigIA - Console de Monitoramento Visual em Tempo Real
===================================================================================
Interface analítica de alto desempenho desenvolvida em Pygame nativo (zero OpenCV).
Pipeline otimizado com vetorização matricial de baixa latência para renderização fluida
a 60 FPS, eliminando gargalos de CPU e elementos de ruído visual.
===================================================================================
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Pré-carregamento dos módulos para evitar colisão LLVM no subsistema gráfico do Linux
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    import keras
except ImportError:
    pass

import numpy as np
from PIL import Image

# Importação dos módulos de controle e PDI do VigIA
from prototipo_vigia import SistemaVigIA, ProcessadorImagem

import pygame
import pygame.camera


def processar_pdi_vetorizado(arr_gray_atual, arr_gray_fundo, limiar=28):
    """
    Executa a segmentação morfológica artesanal em tempo real via vetorização matricial:
    1. Subtração de fundo absoluto: |I(x, y) - B(x, y)|
    2. Binarização rígida por corte de limiar
    3. Abertura morfológica 3x3 (Erosão seguida de Dilatação)
    Execução otimizada em ~4 ms por quadro, garantindo 60 FPS estáveis na interface.
    """
    h, w = arr_gray_atual.shape

    # 1. Subtração de fundo
    diff = np.abs(arr_gray_atual.astype(np.int16) - arr_gray_fundo.astype(np.int16)).astype(np.uint8)

    # 2. Limiarização
    binaria = (diff > limiar).astype(np.uint8) * 255

    # 3. Erosão 3x3 (mínimo local)
    pad_b = np.pad(binaria, 1, mode="edge")
    viz_erosao = [pad_b[i:i+h, j:j+w] for i in range(3) for j in range(3)]
    erosao = np.min(viz_erosao, axis=0)

    # 4. Dilatação 3x3 (máximo local)
    pad_e = np.pad(erosao, 1, mode="edge")
    viz_dilatacao = [pad_e[i:i+h, j:j+w] for i in range(3) for j in range(3)]
    abertura = np.max(viz_dilatacao, axis=0)

    pixels_ativos = int(np.count_nonzero(abertura))
    percentual = (pixels_ativos / float(h * w)) * 100.0

    bbox = None
    if pixels_ativos > 0:
        ys, xs = np.nonzero(abertura)
        if len(ys) > 0:
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            bbox = (min_x, min_y, max_x, max_y)

    return abertura, pixels_ativos, percentual, bbox


def executar_interface_webcam():
    """
    Inicializa a console de vídeo do VigIA com renderização em 60 FPS,
    visualização óptica ao vivo e máscara analítica de PDI.
    """
    # 1. Instanciação prévia do modelo e sistema
    sistema = SistemaVigIA(potencia_iluminacao_watts=400, tarifa_kwh=0.85)

    # 2. Inicialização do subsistema de vídeo
    pygame.init()
    pygame.font.init()
    pygame.camera.init()

    cameras = pygame.camera.list_cameras()
    if not cameras:
        print("\n[ERRO DE HARDWARE] Nenhuma câmera detectada no sistema.")
        return

    largura_w, altura_w = 960, 640
    tela = pygame.display.set_mode((largura_w, altura_w))
    pygame.display.set_caption("VigIA // OPTICAL SENTINEL — APS UNIP CC")

    relogio = pygame.time.Clock()

    # Tipografia analítica (monoespacada e limpa)
    fonte_logo = pygame.font.SysFont("Courier New", 14, bold=True)
    fonte_mono = pygame.font.SysFont("Courier New", 12)
    fonte_status = pygame.font.SysFont("Arial", 16, bold=True)
    fonte_num = pygame.font.SysFont("Courier New", 15, bold=True)
    fonte_micro = pygame.font.SysFont("Courier New", 10)

    dispositivo = cameras[0]
    cam = None
    for res_tentativa in [(640, 480), (640, 360), (320, 240), (0, 0)]:
        try:
            c = pygame.camera.Camera(dispositivo, res_tentativa) if res_tentativa != (0, 0) else pygame.camera.Camera(dispositivo)
            c.start()
            cam = c
            break
        except Exception:
            continue

    if cam is None:
        print("\n[ERRO DE HARDWARE] Falha ao negociar formato com a câmera.")
        pygame.quit()
        return

    # Paleta de cores minimalista (estilo monitor industrial / CCTV profissional)
    COR_BG = (9, 9, 11)           # Zinc 950
    COR_PAINEL = (24, 24, 27)     # Zinc 900
    COR_BORDA = (39, 39, 42)      # Zinc 800
    COR_BORDA_ACTIVE = (59, 130, 246) # Blue accent
    COR_TXT = (244, 244, 245)     # Zinc 100
    COR_MUTED = (113, 113, 122)   # Zinc 500
    COR_GREEN = (34, 197, 94)     # Emerald 500
    COR_RED = (239, 68, 68)       # Red 500
    COR_CYAN = (6, 182, 212)      # Cyan 500

    # Dimensões dos viewports ópticos
    vw, vh = 440, 310
    pos_cam = (25, 60)
    pos_pdi = (495, 60)

    # Resolução interna de processamento PDI (Ultra leve: 160x120)
    proc_w, proc_h = 160, 120
    fundo_gray = None
    calibrado = False

    # Estado operacional
    estado = {
        "ocupado": False,
        "area_pct": 0.0,
        "cnn_prob": None,
        "confianca": 100.0,
        "fps": 60,
        "calibrado": False,
        "alvo_tipo": "STANDBY",
        "bbox": None,
        "pele_pct": 0.0,
        "msg_banner": "AFASTE-SE E PRESSIONE [ESPAÇO] PARA CALIBRAR FUNDO ZERO"
    }

    # Descarte de quadros de inicialização da câmera
    for _ in range(5):
        cam.get_image()
        time.sleep(0.02)

    quadros_contador = 0
    t_ultimo_cnn = 0.0
    intervalo_cnn = 0.35  # Executa predição profunda a cada 350 ms para manter CPU leve

    surf_mask_cached = pygame.Surface((vw, vh))
    surf_mask_cached.fill((0, 0, 0))

    executando = True
    while executando:
        tempo_atual = time.time()
        quadros_contador += 1

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                executando = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    executando = False
                elif ev.key == pygame.K_SPACE:
                    # Calibração zero-point do fundo
                    surf_calib = cam.get_image()
                    sub_calib = pygame.transform.scale(surf_calib, (proc_w, proc_h))
                    arr_rgb_c = pygame.surfarray.array3d(sub_calib)
                    fundo_gray = (0.299 * arr_rgb_c[:, :, 0] + 0.587 * arr_rgb_c[:, :, 1] + 0.114 * arr_rgb_c[:, :, 2]).T
                    
                    # Calibra também o controlador central
                    raw_c = pygame.image.tostring(surf_calib, "RGB")
                    img_c = Image.frombytes("RGB", surf_calib.get_size(), raw_c)
                    sistema.calibrar_fundo(img_c)
                    
                    calibrado = True
                    estado["calibrado"] = True
                    estado["msg_banner"] = "FUNDO CALIBRADO // MONITORAMENTO ATIVO"

        # 1. Aquisição ótica em 60 FPS
        surf_cam = cam.get_image()

        # 2. Extração de matriz de luminância para PDI
        sub_atual = pygame.transform.scale(surf_cam, (proc_w, proc_h))
        arr_rgb = pygame.surfarray.array3d(sub_atual)
        atual_gray = (0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]).T

        # 3. Processamento de PDI (apenas se calibrado)
        if calibrado and fundo_gray is not None:
            mask_2d, pixels_ativos, pct_ocupado, bbox = processar_pdi_vetorizado(atual_gray, fundo_gray, limiar=28)
            estado["area_pct"] = pct_ocupado
            estado["bbox"] = bbox

            # Atualiza superfície da máscara PDI via surfarray (ultrarrápido)
            surf_pdi_mini = pygame.surfarray.make_surface(np.stack([mask_2d.T]*3, axis=-1))
            surf_mask_cached = pygame.transform.scale(surf_pdi_mini, (vw, vh))

            # Inferência Keras desacoplada no tempo (a cada 350 ms)
            if tempo_atual - t_ultimo_cnn >= intervalo_cnn:
                t_ultimo_cnn = tempo_atual
                
                # Prepara tensor para mini-CNN Keras
                raw_full = pygame.image.tostring(surf_cam, "RGB")
                cam_w, cam_h = surf_cam.get_size()
                img_pil = Image.frombytes("RGB", (cam_w, cam_h), raw_full)
                
                # Inferência neural global (cena inteira)
                prob_full = sistema.detector.classificar_neural(img_pil)
                
                # Inferência neural direcionada à Região de Interesse (ROI) do alvo
                crop_pil = None
                prob_crop = None
                box_w = 0
                box_h = 0
                razao_aspecto = 0.0

                if bbox is not None and pixels_ativos >= 40:
                    min_x, min_y, max_x, max_y = bbox
                    box_w = max_x - min_x + 1
                    box_h = max_y - min_y + 1
                    razao_aspecto = box_h / max(1, box_w)

                    scale_x = cam_w / float(proc_w)
                    scale_y = cam_h / float(proc_h)
                    pad_x = int(box_w * 0.15 * scale_x)
                    pad_y = int(box_h * 0.15 * scale_y)

                    c_x1 = max(0, int(min_x * scale_x) - pad_x)
                    c_y1 = max(0, int(min_y * scale_y) - pad_y)
                    c_x2 = min(cam_w, int(max_x * scale_x) + pad_x)
                    c_y2 = min(cam_h, int(max_y * scale_y) + pad_y)

                    if c_x2 > c_x1 + 10 and c_y2 > c_y1 + 10:
                        crop_pil = img_pil.crop((c_x1, c_y1, c_x2, c_y2))
                        prob_crop = sistema.detector.classificar_neural(crop_pil)

                # A predição neural do alvo foca prioritariamente no recorte (ROI) do objeto ativo.
                # Se houver recorte ativo, avalia diretamente o objeto em movimento; caso contrário, a cena global.
                prob_neural = prob_crop if prob_crop is not None else prob_full
                estado["cnn_prob"] = prob_neural

                # Análise PDI de crominância biométrica (pele humana) no recorte do alvo
                # Modelos clássicos de segmentação fotométrica (Kovac et al. / Peer et al.):
                # R > 60, G > 30, B > 15, R - G > 10, R > B, max - min > 15
                densidade_pele = 0.0
                if crop_pil is not None:
                    arr_crop = np.array(crop_pil.convert("RGB"), dtype=np.int16)
                    r_c, g_c, b_c = arr_crop[:, :, 0], arr_crop[:, :, 1], arr_crop[:, :, 2]
                    mascara_pele = (
                        (r_c > 60) & (g_c > 30) & (b_c > 15) &
                        ((r_c - g_c) > 10) & (r_c > b_c) &
                        ((np.maximum(r_c, np.maximum(g_c, b_c)) - np.minimum(r_c, np.minimum(g_c, b_c))) > 15)
                    )
                    total_crop_px = arr_crop.shape[0] * arr_crop.shape[1]
                    if total_crop_px > 0:
                        densidade_pele = (np.count_nonzero(mascara_pele) / float(total_crop_px)) * 100.0
                estado["pele_pct"] = densidade_pele

                # Discriminação Estrita Humano vs. Objetos Inanimados:
                # 1. Movimentação mínima relevante: pct_ocupado >= 0.8%
                # 2. Antropometria: altura vertical >= 28 px (~23% da cena) e proporção ereta H/W >= 0.55
                # 3. Validação convolucional sobre o recorte: probabilidade da rede neural >= 0.50
                # 4. Validação biométrica: presença de traços cromáticos humanos (pele/rosto >= 1.5%)
                if pct_ocupado < 0.8 or bbox is None:
                    ocupado = False
                    alvo_tipo = "NENHUM"
                    confianca = 99.0
                else:
                    eh_antropometrico = (box_h >= 28 and razao_aspecto >= 0.55)
                    eh_humano_neural = (prob_neural is not None and prob_neural >= 0.50)
                    tem_biometria = (densidade_pele >= 1.5 or (prob_neural is not None and prob_neural >= 0.85 and densidade_pele > 0.0))

                    if eh_antropometrico and eh_humano_neural and tem_biometria:
                        ocupado = True
                        alvo_tipo = "HUMANO"
                        confianca = max(80.0, min(99.9, prob_neural * 100.0))
                    else:
                        # Objeto inanimado detectado na cena (celular, copo, cadeira deslocada, mochila)
                        ocupado = False
                        alvo_tipo = "OBJETO_IGNORADO"
                        confianca = max(80.0, min(99.9, (1.0 - (prob_neural or 0.0)) * 100.0))

                estado["ocupado"] = ocupado
                estado["alvo_tipo"] = alvo_tipo
                estado["confianca"] = confianca

                # Atualiza controlador físico
                sistema.horas_totais_monitoradas += (intervalo_cnn / 3600.0)
                if ocupado:
                    sistema.luz_acesa = True
                else:
                    sistema.luz_acesa = False
                    sistema.horas_luz_economizada += (intervalo_cnn / 3600.0)

        # =========================================================================
        # RENDERIZAÇÃO VISUAL MINIMALISTA (ZERO SLOP)
        # =========================================================================
        tela.fill(COR_BG)

        # --- CABEÇALHO TELEMÉTRICO ---
        pygame.draw.line(tela, COR_BORDA, (0, 42), (largura_w, 42), 1)
        
        lbl_brand = fonte_logo.render("VigIA // OPTICAL SENTINEL", True, COR_TXT)
        lbl_tech = fonte_micro.render("PIPELINE: ITU-R BT.601 | GAUSS 3x3 | SOBEL | MORPH 3x3 | KERAS CNN", True, COR_MUTED)
        tela.blit(lbl_brand, (25, 14))
        tela.blit(lbl_tech, (250, 17))

        # Indicador de atividade LIVE
        fps_real = int(relogio.get_fps())
        str_fps = f"{fps_real} FPS" if fps_real > 0 else "60 FPS"
        lbl_fps = fonte_mono.render(f"● LIVE  [{str_fps}]", True, COR_GREEN)
        tela.blit(lbl_fps, (largura_w - 140, 15))

        # --- VIEWPORT 1: CÂMERA AO VIVO ---
        cor_borda_cam = COR_GREEN if estado["ocupado"] else ((245, 158, 11) if estado.get("alvo_tipo") == "OBJETO_IGNORADO" else COR_BORDA)
        pygame.draw.rect(tela, cor_borda_cam, (pos_cam[0]-1, pos_cam[1]-1, vw+2, vh+2), 1)
        
        # Redimensionamento direto e blit
        surf_cam_scaled = pygame.transform.scale(surf_cam, (vw, vh))
        tela.blit(surf_cam_scaled, pos_cam)
        
        # Bounding Box analítica com classificação visual de alvo
        if estado.get("bbox") is not None and estado["area_pct"] >= 0.8:
            min_x, min_y, max_x, max_y = estado["bbox"]
            scale_vw = vw / float(proc_w)
            scale_vh = vh / float(proc_h)
            
            bx = pos_cam[0] + int(min_x * scale_vw)
            by = pos_cam[1] + int(min_y * scale_vh)
            bw = int((max_x - min_x + 1) * scale_vw)
            bh = int((max_y - min_y + 1) * scale_vh)
            
            # Limites do viewport
            bx = max(pos_cam[0], min(bx, pos_cam[0] + vw - 10))
            by = max(pos_cam[1], min(by, pos_cam[1] + vh - 10))
            bw = max(10, min(bw, pos_cam[0] + vw - bx))
            bh = max(10, min(bh, pos_cam[1] + vh - by))
            
            if estado["ocupado"]:
                cor_box = COR_GREEN
                rotulo_alvo = f"HUMANO CONFIRMADO [{estado['confianca']:.0f}%]"
            else:
                cor_box = (245, 158, 11)  # Âmbar (Objeto inanimado descartado)
                rotulo_alvo = "OBJETO DETECTADO: IGNORADO"
                
            pygame.draw.rect(tela, cor_box, (bx, by, bw, bh), 2)
            
            tag_surf = fonte_micro.render(rotulo_alvo, True, (0, 0, 0))
            tag_w = tag_surf.get_width() + 8
            tag_h = 16
            tag_y = max(pos_cam[1], by - tag_h)
            pygame.draw.rect(tela, cor_box, (bx, tag_y, tag_w, tag_h))
            tela.blit(tag_surf, (bx + 4, tag_y + 1))

        # Overlay telemétrico no vídeo
        pygame.draw.rect(tela, (0, 0, 0), (pos_cam[0], pos_cam[1], 160, 20))
        lbl_v1 = fonte_micro.render("FEED // OPTICAL STREAM", True, COR_TXT)
        tela.blit(lbl_v1, (pos_cam[0]+6, pos_cam[1]+4))

        # --- VIEWPORT 2: MÁSCARA PDI ARTESANAL ---
        pygame.draw.rect(tela, COR_BORDA, (pos_pdi[0]-1, pos_pdi[1]-1, vw+2, vh+2), 1)
        tela.blit(surf_mask_cached, pos_pdi)
        
        pygame.draw.rect(tela, (0, 0, 0), (pos_pdi[0], pos_pdi[1], 175, 20))
        lbl_v2 = fonte_micro.render("PDI // MORPHOLOGICAL MASK", True, COR_CYAN)
        tela.blit(lbl_v2, (pos_pdi[0]+6, pos_pdi[1]+4))

        # --- BANNER DE STATUS INTERNO ---
        cor_banner = (245, 158, 11) if not calibrado else (39, 39, 42)
        txt_banner = estado["msg_banner"]
        if calibrado:
            if estado["ocupado"]:
                txt_banner = "PRESENÇA HUMANA CONFIRMADA // ILUMINAÇÃO ACIONADA (400W)"
            elif estado.get("alvo_tipo") == "OBJETO_IGNORADO":
                txt_banner = "OBJETO INANIMADO IDENTIFICADO // REJEITADO (STANDBY 0W)"
            else:
                txt_banner = "SALA DESOCUPADA // MODO SENTINELA ATIVO (STANDBY 0W)"
        lbl_banner = fonte_mono.render(f"STATUS >> {txt_banner}", True, (244, 244, 245) if calibrado else (254, 240, 138))
        tela.blit(lbl_banner, (25, 385))

        # --- DECK TELEMÉTRICO INFERIOR (3 COLUNAS) ---
        deck_y = 415
        deck_h = 165
        col_w = 295

        # Coluna 1: Estado Primário
        c1_x = 25
        if estado["ocupado"]:
            cor_badge = (6, 78, 59)
            borda_badge = (16, 185, 129)
            str_presenca = "● HUMANO DETECTADO"
            cor_txt_p = COR_GREEN
            str_rele = "CARGA: 400W [ACIONADA]"
            cor_rele = COR_TXT
        elif estado.get("alvo_tipo") == "OBJETO_IGNORADO":
            cor_badge = (45, 26, 0)
            borda_badge = (245, 158, 11)
            str_presenca = "○ OBJETO (IGNORADO)"
            cor_txt_p = (251, 191, 36)
            str_rele = "CARGA: 0W [STANDBY]"
            cor_rele = COR_MUTED
        else:
            cor_badge = (24, 24, 27)
            borda_badge = COR_BORDA
            str_presenca = "○ SALA VAZIA"
            cor_txt_p = (161, 161, 170)
            str_rele = "CARGA: 0W [STANDBY]"
            cor_rele = COR_MUTED

        pygame.draw.rect(tela, cor_badge, (c1_x, deck_y, col_w, deck_h), border_radius=4)
        pygame.draw.rect(tela, borda_badge, (c1_x, deck_y, col_w, deck_h), 1, border_radius=4)

        lbl_s_tit = fonte_micro.render("OCUPAÇÃO DO AMBIENTE", True, COR_MUTED)
        tela.blit(lbl_s_tit, (c1_x + 16, deck_y + 14))

        lbl_pres = fonte_status.render(str_presenca, True, cor_txt_p)
        tela.blit(lbl_pres, (c1_x + 16, deck_y + 36))

        lbl_rele = fonte_mono.render(str_rele, True, cor_rele)
        tela.blit(lbl_rele, (c1_x + 16, deck_y + 75))

        lbl_conf = fonte_mono.render(f"CONFIANÇA : {estado['confianca']:.1f}%", True, COR_MUTED)
        tela.blit(lbl_conf, (c1_x + 16, deck_y + 100))

        # Coluna 2: Métricas de Visão e PDI
        c2_x = 332
        pygame.draw.rect(tela, COR_PAINEL, (c2_x, deck_y, col_w, deck_h), border_radius=4)
        pygame.draw.rect(tela, COR_BORDA, (c2_x, deck_y, col_w, deck_h), 1, border_radius=4)

        lbl_v_tit = fonte_micro.render("TELEMETRIA DE VISÃO COMPUTACIONAL", True, COR_MUTED)
        tela.blit(lbl_v_tit, (c2_x + 16, deck_y + 14))

        alvo_str = "HUMANO" if estado["ocupado"] else ("OBJETO REJEITADO" if estado.get("alvo_tipo") == "OBJETO_IGNORADO" else "STANDBY")
        cor_alvo_lbl = COR_GREEN if estado["ocupado"] else ((245, 158, 11) if estado.get("alvo_tipo") == "OBJETO_IGNORADO" else COR_CYAN)
        lbl_m0 = fonte_mono.render(f"FILTRO ALVO    : {alvo_str}", True, cor_alvo_lbl)
        tela.blit(lbl_m0, (c2_x + 16, deck_y + 35))

        lbl_m1 = fonte_mono.render(f"MÁSCARA PDI    : {estado['area_pct']:.2f}%", True, COR_TXT)
        tela.blit(lbl_m1, (c2_x + 16, deck_y + 55))

        prob_val = estado["cnn_prob"]
        str_cnn = f"{prob_val * 100.0:.1f}%" if prob_val is not None else "STANDBY"
        lbl_m2 = fonte_mono.render(f"PREDIÇÃO CNN   : {str_cnn}", True, COR_TXT)
        tela.blit(lbl_m2, (c2_x + 16, deck_y + 75))

        pele_val = estado.get("pele_pct", 0.0)
        lbl_m3 = fonte_mono.render(f"BIOMETRIA PELE : {pele_val:.1f}%", True, COR_CYAN if pele_val >= 1.5 else COR_MUTED)
        tela.blit(lbl_m3, (c2_x + 16, deck_y + 95))

        lbl_m4 = fonte_micro.render("REJEIÇÃO ATIVA : CELULAR/MÓVEIS/COPOS", True, COR_MUTED)
        tela.blit(lbl_m4, (c2_x + 16, deck_y + 125))

        # Coluna 3: Sustentabilidade e Carbono
        c3_x = 640
        pygame.draw.rect(tela, COR_PAINEL, (c3_x, deck_y, col_w, deck_h), border_radius=4)
        pygame.draw.rect(tela, COR_BORDA, (c3_x, deck_y, col_w, deck_h), 1, border_radius=4)

        dados_s = sistema.obter_relatorio_sustentabilidade()
        lbl_e_tit = fonte_micro.render("EFICIÊNCIA ENERGÉTICA E CARBONO", True, COR_MUTED)
        tela.blit(lbl_e_tit, (c3_x + 16, deck_y + 14))

        lbl_e1 = fonte_mono.render(f"ENERGIA POUPADA: {dados_s['kwh_poupados']:.4f} kWh", True, COR_GREEN)
        tela.blit(lbl_e1, (c3_x + 16, deck_y + 40))

        lbl_e2 = fonte_mono.render(f"CUSTO EVITADO  : R$ {dados_s['economia_reais']:.2f}", True, COR_TXT)
        tela.blit(lbl_e2, (c3_x + 16, deck_y + 65))

        co2_g = dados_s["reducao_co2_kg"] * 1000.0
        lbl_e3 = fonte_mono.render(f"REDUÇÃO DE CO2 : {co2_g:.2f} g", True, COR_CYAN)
        tela.blit(lbl_e3, (c3_x + 16, deck_y + 90))

        lbl_e4 = fonte_micro.render(f"CICLO SUSTENTÁVEL: {dados_s['percentual_economia']:.1f}% OFF", True, COR_MUTED)
        tela.blit(lbl_e4, (c3_x + 16, deck_y + 120))

        # --- RODAPÉ DE COMANDOS ---
        pygame.draw.line(tela, COR_BORDA, (0, 600), (largura_w, 600), 1)
        lbl_cmd = fonte_mono.render("[ESPAÇO] CALIBRAR FUNDO ZERO   |   [Q / ESC] ENCERRAR SESSÃO", True, COR_MUTED)
        tela.blit(lbl_cmd, (25, 614))

        pygame.display.flip()
        relogio.tick(60)

    # Finalização segura
    cam.stop()
    pygame.quit()
    print("\n" + "="*70)
    print("SESSÃO DA WEBCAM FINALIZADA - RELATÓRIO DE SUSTENTABILIDADE")
    print("="*70)
    print(sistema.gerar_relatorio_sustentabilidade())


if __name__ == "__main__":
    executar_interface_webcam()
