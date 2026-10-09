"""Modo experimental de controle do cursor com MobileGaze (rede neural leve de gaze, ONNX).

Em vez de calcular a posicao da iris dentro do olho (metodo geometrico usado em olho.py),
esse script usa uma rede neural leve (MobileOne S0, do repositorio yakhyo/gaze-estimation,
MIT license) para prever o angulo do olhar (pitch/yaw) direto da imagem do rosto. Piscar e
sobrancelha continuam vindo do MediaPipe, que ja roda de qualquer forma para achar o rosto.

A classe GazeEstimationONNX abaixo (marcada no comentario) e adaptada do arquivo
onnx_inference.py desse repositorio (https://github.com/yakhyo/gaze-estimation), mantendo
a logica de pre/pos-processamento identica ao original — so a deteccao de rosto foi
trocada pela do MediaPipe (que ja estava rodando por causa do piscar/sobrancelha), evitando
rodar dois detectores de rosto por frame.

ANTES de calibrar (16 pontos, alguns minutos), rode o benchmark para saber se essa rede
roda rapido o suficiente na sua CPU:

    python olho_gaze_onnx.py --benchmark

Instalar as dependencias extras (leve, so onnxruntime, sem torch):
    pip install -r requirements-gaze-onnx.txt

Baixar o modelo (oficial, GitHub Releases, ~5MB):
    python download_gaze_model.py

Rodar de verdade (calibracao + cursor):
    python olho_gaze_onnx.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Tuple

import olho as base

CAMINHO_MODELO_GAZE = Path(__file__).with_name("models") / "mobileone_s0_gaze.onnx"
MARGEM_RECORTE_ROSTO = 0.25  # extra em torno do bbox apertado dos landmarks, em fracao da largura/altura.


class GazeEstimationONNX:
    """Adaptado de onnx_inference.py em github.com/yakhyo/gaze-estimation (MIT license).

    Copyright (c) 2024 Yakhyokhuja Valikhujaev. Logica de pre-processamento (resize
    448x448, normalizacao ImageNet) e pos-processamento (90 bins de 4 graus, softmax,
    offset de -180) mantida identica ao original; so a origem do recorte de rosto muda
    (aqui vem do MediaPipe, la vinha do RetinaFace via uniface).
    """

    def __init__(self, cv2, np, ort, model_path: str) -> None:
        self._cv2 = cv2
        self._np = np
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = tuple(input_cfg.shape[2:][::-1])  # (largura, altura)

        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]
        assert len(self.output_names) == 2, f"Esperava 2 saidas do modelo, veio {len(self.output_names)}"

    def preprocess(self, imagem_rgb):
        np = self._np
        cv2 = self._cv2
        imagem = cv2.resize(imagem_rgb, self.input_size)
        imagem = imagem.astype(np.float32) / 255.0
        imagem = (imagem - self.input_mean) / self.input_std
        imagem = np.transpose(imagem, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(imagem, axis=0).astype(np.float32)  # CHW -> BCHW

    def softmax(self, x):
        np = self._np
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def decode(self, yaw_logits, pitch_logits) -> Tuple[float, float]:
        np = self._np
        yaw_probs = self.softmax(yaw_logits)
        pitch_probs = self.softmax(pitch_logits)
        yaw = np.sum(yaw_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        pitch = np.sum(pitch_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        return np.radians(yaw[0]), np.radians(pitch[0])

    def estimate(self, imagem_rgb) -> Tuple[float, float]:
        """Recebe um recorte de rosto em RGB (qualquer tamanho). Retorna (yaw, pitch) em radianos."""
        entrada = self.preprocess(imagem_rgb)
        saidas = self.session.run(self.output_names, {self.input_name: entrada})
        return self.decode(saidas[0], saidas[1])


def carregar_modelo_gaze(cv2, np):
    if not CAMINHO_MODELO_GAZE.exists():
        raise FileNotFoundError(
            "Modelo de gaze nao encontrado. Rode: "
            ".\\.venv\\Scripts\\python.exe download_gaze_model.py"
        )

    print("Carregando ONNX Runtime e o modelo de gaze...", flush=True)
    import onnxruntime as ort

    return GazeEstimationONNX(cv2, np, ort, str(CAMINHO_MODELO_GAZE))


def calcular_bbox_do_rosto(pontos_rosto, largura: int, altura: int, margem: float = MARGEM_RECORTE_ROSTO):
    xs = [ponto.x for ponto in pontos_rosto]
    ys = [ponto.y for ponto in pontos_rosto]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    largura_bbox = x_max - x_min
    altura_bbox = y_max - y_min
    x_min -= largura_bbox * margem
    x_max += largura_bbox * margem
    y_min -= altura_bbox * margem
    y_max += altura_bbox * margem

    x_min_px = max(int(x_min * largura), 0)
    y_min_px = max(int(y_min * altura), 0)
    x_max_px = min(int(x_max * largura), largura)
    y_max_px = min(int(y_max * altura), altura)
    return x_min_px, y_min_px, x_max_px, y_max_px


def recortar_rosto_rgb(cv2, quadro_bgr, bbox):
    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        return None
    recorte = quadro_bgr[y_min:y_max, x_min:x_max]
    return cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)


def calcular_features_gaze(modelo_gaze, cv2, quadro_bgr, pontos_rosto):
    """Retorna (yaw, pitch) em radianos, ou None se o recorte for invalido."""
    altura, largura, _ = quadro_bgr.shape
    bbox = calcular_bbox_do_rosto(pontos_rosto, largura, altura)
    recorte_rgb = recortar_rosto_rgb(cv2, quadro_bgr, bbox)
    if recorte_rgb is None or recorte_rgb.size == 0:
        return None
    return modelo_gaze.estimate(recorte_rgb)


def rodar_benchmark(n_frames: int = 60) -> None:
    (cv2, np, mp, BaseOptions, FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections, RunningMode, _) = (
        base.carregar_dependencias()
    )
    modelo_gaze = carregar_modelo_gaze(cv2, np)

    config = base.Configuracoes()
    camera = cv2.VideoCapture(config.indice_camera)
    if not camera.isOpened():
        raise RuntimeError("Nao foi possivel abrir a camera.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.largura_camera)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.altura_camera)
    camera.set(cv2.CAP_PROP_FPS, config.fps_camera)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector_rosto = base.criar_detector_de_rosto(
        BaseOptions, FaceLandmarker, FaceLandmarkerOptions, RunningMode, config
    )
    proximo_tempo_ms = base.criar_gerador_de_timestamps()

    print(f"Rodando benchmark com {n_frames} frames (precisa do seu rosto na camera)...", flush=True)
    tempos_mediapipe_ms = []
    tempos_gaze_ms = []
    frames_com_rosto = 0

    try:
        while frames_com_rosto < n_frames:
            resultado, quadro = base.capturar_quadro_processado(cv2, mp, camera, detector_rosto, proximo_tempo_ms)
            if quadro is None:
                break
            if not resultado.face_landmarks:
                continue

            pontos_rosto = resultado.face_landmarks[0]
            altura, largura, _ = quadro.shape

            inicio_mediapipe = time.monotonic()
            bbox = calcular_bbox_do_rosto(pontos_rosto, largura, altura)
            recorte_rgb = recortar_rosto_rgb(cv2, quadro, bbox)
            tempos_mediapipe_ms.append((time.monotonic() - inicio_mediapipe) * 1000)

            if recorte_rgb is None:
                continue

            inicio_gaze = time.monotonic()
            modelo_gaze.estimate(recorte_rgb)
            tempos_gaze_ms.append((time.monotonic() - inicio_gaze) * 1000)

            frames_com_rosto += 1
            if frames_com_rosto % 10 == 0:
                print(f"  {frames_com_rosto}/{n_frames} frames processados...", flush=True)
    finally:
        detector_rosto.close()
        camera.release()

    if not tempos_gaze_ms:
        print("Nao consegui coletar nenhum frame com rosto detectado. Verifique a camera/iluminacao.")
        return

    media_mediapipe = statistics.mean(tempos_mediapipe_ms)
    media_gaze = statistics.mean(tempos_gaze_ms)
    media_total = media_mediapipe + media_gaze
    fps_estimado = 1000 / media_total if media_total > 0 else float("inf")

    print("\n--- Resultado do benchmark ---")
    print(f"Recorte do rosto (MediaPipe + crop): {media_mediapipe:.1f} ms/frame em media")
    print(f"Inferencia do modelo de gaze (CPU):   {media_gaze:.1f} ms/frame em media")
    print(f"Total: {media_total:.1f} ms/frame  (~{fps_estimado:.1f} fps possiveis)")
    print("\nPara controle de cursor responsivo, o ideal e ficar por volta de 30fps (33ms/frame).")
    if media_total > 60:
        print("Isso esta bem mais lento que 30fps — o cursor provavelmente vai parecer com atraso.")
    elif media_total > 33:
        print("Um pouco mais lento que 30fps, mas deve ser usavel com mais suavizacao/atraso.")
    else:
        print("Dentro do orcamento de 30fps — deve ficar responsivo. Esperado, e um modelo bem leve.")
    print("--- Fim do benchmark ---\n")


PONTOS_DE_CALIBRACAO = base.PONTOS_DE_CALIBRACAO
AMOSTRAS_MINIMAS_POR_PONTO = base.AMOSTRAS_MINIMAS_POR_PONTO
AMOSTRAS_MAXIMAS_POR_PONTO = base.AMOSTRAS_MAXIMAS_POR_PONTO
DESVIO_MAXIMO_ESTAVEL = 0.175  # ~10 graus. Medido na pratica: mesmo olhando fixo, o
# desvio do modelo entre frames fica na faixa de 2 a 10 graus (bem mais ruidoso que o
# metodo geometrico da iris). Um limite de 2 graus (usado antes) nunca era atingido.
JANELA_CALIBRACAO = "Calibracao do olhar (MobileGaze)"


def _calcular_desvio(leituras):
    """Retorna (desvio_yaw, desvio_pitch) das ultimas amostras, ou None se ainda nao ha o suficiente."""
    if len(leituras) < AMOSTRAS_MINIMAS_POR_PONTO:
        return None
    recentes = leituras[-AMOSTRAS_MINIMAS_POR_PONTO:]
    yaws = [amostra[0] for amostra in recentes]
    pitches = [amostra[1] for amostra in recentes]
    return statistics.pstdev(yaws), statistics.pstdev(pitches)


def _esta_estavel(leituras) -> bool:
    desvio = _calcular_desvio(leituras)
    if desvio is None:
        return False
    desvio_yaw, desvio_pitch = desvio
    return desvio_yaw < DESVIO_MAXIMO_ESTAVEL and desvio_pitch < DESVIO_MAXIMO_ESTAVEL


def calibrar_regressao_gaze(cv2, np, mp, camera, detector_rosto, modelo_gaze, proximo_tempo_ms):
    largura_janela, altura_janela = 1280, 720
    cv2.namedWindow(JANELA_CALIBRACAO, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(JANELA_CALIBRACAO, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        cv2.resizeWindow(JANELA_CALIBRACAO, largura_janela, altura_janela)

    linhas = []
    alvos_x = []
    alvos_y = []

    for indice_ponto, (rel_x, rel_y) in enumerate(PONTOS_DE_CALIBRACAO):
        leituras = []
        while True:
            resultado, quadro = base.capturar_quadro_processado(cv2, mp, camera, detector_rosto, proximo_tempo_ms)
            if quadro is None:
                continue

            if resultado.face_landmarks:
                features = calcular_features_gaze(modelo_gaze, cv2, quadro, resultado.face_landmarks[0])
                if features is not None:
                    leituras.append(features)
                    if len(leituras) > AMOSTRAS_MAXIMAS_POR_PONTO:
                        leituras.pop(0)

            desvio = _calcular_desvio(leituras)
            pronto = desvio is not None and desvio[0] < DESVIO_MAXIMO_ESTAVEL and desvio[1] < DESVIO_MAXIMO_ESTAVEL

            canvas = np.zeros((altura_janela, largura_janela, 3), dtype=np.uint8)
            ponto = (int(rel_x * largura_janela), int(rel_y * altura_janela))
            cor_ponto = (60, 220, 60) if pronto else (60, 60, 255)
            cv2.circle(canvas, ponto, 22, (255, 255, 255), -1)
            cv2.circle(canvas, ponto, 8, cor_ponto, -1)
            cv2.putText(
                canvas,
                f"Ponto {indice_ponto + 1}/{len(PONTOS_DE_CALIBRACAO)}: olhe fixo e pressione ESPACO",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            if desvio is None:
                situacao = f"coletando leituras... ({len(leituras)}/{AMOSTRAS_MINIMAS_POR_PONTO})"
            else:
                desvio_graus = max(np.degrees(desvio[0]), np.degrees(desvio[1]))
                situacao = "pronto" if pronto else f"estabilizando... (desvio {desvio_graus:.1f} graus)"
            cv2.putText(canvas, situacao, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(
                canvas,
                "Pressione 's' para pular a calibracao",
                (40, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            cv2.imshow(JANELA_CALIBRACAO, canvas)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("s"):
                cv2.destroyWindow(JANELA_CALIBRACAO)
                return None
            if tecla == 32 and pronto:
                amostras_recentes = leituras[-AMOSTRAS_MINIMAS_POR_PONTO:]
                mediana = tuple(statistics.median(valor) for valor in zip(*amostras_recentes))
                linhas.append((*mediana, 1.0))
                alvos_x.append(rel_x)
                alvos_y.append(rel_y)
                break

    cv2.destroyWindow(JANELA_CALIBRACAO)
    pesos_x, pesos_y = base.ajustar_regressao_de_olhar(np, linhas, alvos_x, alvos_y)
    print("pesos_x (yaw, pitch, bias):", [round(float(p), 3) for p in pesos_x])
    print("pesos_y (yaw, pitch, bias):", [round(float(p), 3) for p in pesos_y])
    return pesos_x, pesos_y


def criar_calculador_de_alvo_gaze(modelo_gaze, cv2, pesos_x, pesos_y, tela_esquerda, tela_topo, largura_tela, altura_tela):
    def calcular(quadro, pontos_rosto):
        features = calcular_features_gaze(modelo_gaze, cv2, quadro, pontos_rosto)
        if features is None:
            return None
        vetor = (*features, 1.0)
        rel_x = base.limitar(sum(f * p for f, p in zip(vetor, pesos_x)), 0.0, 1.0)
        rel_y = base.limitar(sum(f * p for f, p in zip(vetor, pesos_y)), 0.0, 1.0)
        x = tela_esquerda + rel_x * (largura_tela - 1)
        y = tela_topo + rel_y * (altura_tela - 1)
        return x, y

    return calcular


def principal() -> None:
    (cv2, np, mp, BaseOptions, FaceLandmarker, FaceLandmarkerOptions, FaceLandmarksConnections, RunningMode, pyautogui) = (
        base.carregar_dependencias()
    )
    modelo_gaze = carregar_modelo_gaze(cv2, np)

    config = base.Configuracoes()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0
    tela_esquerda, tela_topo, largura_tela, altura_tela = base.obter_limites_da_tela(pyautogui)

    camera = cv2.VideoCapture(config.indice_camera)
    if not camera.isOpened():
        raise RuntimeError("Nao foi possivel abrir a camera.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.largura_camera)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.altura_camera)
    camera.set(cv2.CAP_PROP_FPS, config.fps_camera)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(base.TITULO_JANELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(base.TITULO_JANELA, config.largura_previa, 720)
    try:
        cv2.setWindowProperty(base.TITULO_JANELA, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    detector_rosto = base.criar_detector_de_rosto(
        BaseOptions, FaceLandmarker, FaceLandmarkerOptions, RunningMode, config
    )
    proximo_tempo_ms = base.criar_gerador_de_timestamps()

    resultado_calibracao = calibrar_regressao_gaze(cv2, np, mp, camera, detector_rosto, modelo_gaze, proximo_tempo_ms)
    if resultado_calibracao is None:
        print("Calibracao pulada — sem calibracao nao ha mapeamento razoavel yaw/pitch -> tela neste modo.")
        detector_rosto.close()
        camera.release()
        cv2.destroyAllWindows()
        return

    pesos_x, pesos_y = resultado_calibracao
    calcular_alvo = criar_calculador_de_alvo_gaze(
        modelo_gaze, cv2, pesos_x, pesos_y, tela_esquerda, tela_topo, largura_tela, altura_tela
    )

    mouse_x = tela_esquerda + (largura_tela / 2)
    mouse_y = tela_topo + (altura_tela / 2)
    suavizador = base.SuavizadorDeOlhar(config, mouse_x, mouse_y)
    sobreposicao = base.SobreposicaoDeCursor(config.raio_do_cursor)
    olhos_estavam_fechados = False
    inicio_piscada: float | None = None
    mouse_esta_pressionado = False

    try:
        while True:
            resultado, quadro = base.capturar_quadro_processado(cv2, mp, camera, detector_rosto, proximo_tempo_ms)
            if quadro is None:
                break

            if resultado.face_landmarks:
                pontos_rosto = resultado.face_landmarks[0]
                blendshapes = resultado.face_blendshapes[0] if resultado.face_blendshapes else []

                piscar_esquerdo = base.obter_pontuacao_blendshape(blendshapes, "eyeBlinkLeft")
                piscar_direito = base.obter_pontuacao_blendshape(blendshapes, "eyeBlinkRight")
                sobrancelha = base.obter_pontuacao_blendshape(blendshapes, "browInnerUp")

                olhos_fechados = piscar_esquerdo > config.limite_piscar and piscar_direito > config.limite_piscar
                sobrancelha_levantada = sobrancelha > config.limite_sobrancelha

                if olhos_fechados:
                    if inicio_piscada is None:
                        inicio_piscada = time.monotonic()
                    olhos_estavam_fechados = True
                else:
                    if olhos_estavam_fechados and inicio_piscada is not None:
                        duracao_piscada = time.monotonic() - inicio_piscada
                        if duracao_piscada < config.tempo_clique_curto:
                            pyautogui.click(button="left")
                        else:
                            pyautogui.click(button="right")
                    olhos_estavam_fechados = False
                    inicio_piscada = None

                    if sobrancelha_levantada and not mouse_esta_pressionado:
                        pyautogui.mouseDown(button="left")
                        mouse_esta_pressionado = True
                    elif not sobrancelha_levantada and mouse_esta_pressionado:
                        pyautogui.mouseUp(button="left")
                        mouse_esta_pressionado = False

                    alvo = calcular_alvo(quadro, pontos_rosto)
                    if alvo is not None:
                        mouse_x, mouse_y, deve_mover = suavizador.atualizar(*alvo)
                        if deve_mover:
                            pyautogui.moveTo(mouse_x, mouse_y, duration=0)

                base.desenhar_olhos(cv2, quadro, pontos_rosto)
            else:
                suavizador.esquecer_historico()
                if mouse_esta_pressionado:
                    pyautogui.mouseUp(button="left")
                    mouse_esta_pressionado = False
                    olhos_estavam_fechados = False
                    inicio_piscada = None

            previa = base.redimensionar_previa(cv2, quadro, config.largura_previa)
            cv2.imshow(base.TITULO_JANELA, previa)
            sobreposicao.mover_para(mouse_x, mouse_y)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if mouse_esta_pressionado:
            pyautogui.mouseUp(button="left")
        sobreposicao.fechar()
        detector_rosto.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        rodar_benchmark()
    else:
        principal()
