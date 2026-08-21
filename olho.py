from __future__ import annotations

import ctypes
import math
import os
import statistics
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path

PASTA_MATPLOTLIB = Path(__file__).with_name(".matplotlib")
PASTA_MATPLOTLIB.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PASTA_MATPLOTLIB))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CAMINHO_MODELO = Path(__file__).with_name("models") / "face_landmarker.task"
TITULO_JANELA = "Eye Cursor - Preview"

# Indices dos pontos do FaceMesh usados para estimar o olhar.
OLHO_DIREITO_CANTO_EXTERNO = 33
OLHO_DIREITO_CANTO_INTERNO = 133
OLHO_DIREITO_TOPO = 159
OLHO_DIREITO_BASE = 145
IRIS_DIREITA_CENTRO = 468

OLHO_ESQUERDO_CANTO_INTERNO = 362
OLHO_ESQUERDO_CANTO_EXTERNO = 263
OLHO_ESQUERDO_TOPO = 386
OLHO_ESQUERDO_BASE = 374
IRIS_ESQUERDA_CENTRO = 473

# Pontos usados como referencia da pose da cabeca (compensam parte do erro
# introduzido quando a cabeca se move em vez de so os olhos).
NARIZ_PONTA = 1
ROSTO_TEMPORA_ESQUERDA = 234
ROSTO_TEMPORA_DIREITA = 454
ROSTO_TESTA = 10
ROSTO_QUEIXO = 152


@dataclass(frozen=True)
class Configuracoes:
    indice_camera: int = 0
    limite_piscar: float = 0.5
    limite_sobrancelha: float = 0.4
    tempo_clique_curto: float = 0.35
    confianca_deteccao: float = 0.5
    confianca_rastreamento: float = 0.5
    largura_camera: int = 1280
    altura_camera: int = 720
    fps_camera: int = 30
    largura_previa: int = 960
    controle_esquerda: float = 0.32
    controle_direita: float = 0.68
    controle_topo: float = 0.30
    controle_base: float = 0.70

    # --- Suavizacao do cursor ---
    # Mediana movel: descarta landmarks fora da curva. Mais frames = mais estavel, mais atraso.
    janela_mediana: int = 7
    # One Euro Filter: corte minimo em Hz (menor = mais suave com o olhar parado).
    corte_minimo: float = 0.4
    # Quanto o filtro "abre" quando o olhar se move rapido (maior = menos atraso, mais tremor).
    beta_velocidade: float = 0.006
    # Corte do estimador de velocidade do proprio filtro.
    corte_velocidade: float = 1.0
    # Raio em pixels dentro do qual o cursor simplesmente nao se move.
    zona_morta_px: float = 12.0

    # --- Raio de tolerancia do cursor ---
    # Um circulo vermelho acompanha o cursor na tela, do tamanho deste raio (em pixels).
    # Serve como referencia visual de que a posicao exata do olhar tem uma margem de erro
    # esperada — mira no alvo com essa folga em vez de exigir o pixel exato.
    raio_do_cursor: float = 22.0


class FiltroPassaBaixa:
    """Media exponencial com alpha variavel, base do One Euro Filter."""

    def __init__(self) -> None:
        self._valor: float | None = None

    def aplicar(self, valor: float, alpha: float) -> float:
        if self._valor is None:
            self._valor = valor
        else:
            self._valor = alpha * valor + (1.0 - alpha) * self._valor
        return self._valor

    @property
    def valor(self) -> float | None:
        return self._valor


class FiltroUmEuro:
    """One Euro Filter: suaviza muito quando o sinal esta quase parado e libera quando ele acelera.

    E o filtro usado para ponteiros porque resolve o conflito classico entre tremor e atraso:
    o corte do passa-baixa cresce junto com a velocidade estimada do sinal.
    """

    def __init__(self, corte_minimo: float, beta: float, corte_velocidade: float) -> None:
        self.corte_minimo = corte_minimo
        self.beta = beta
        self.corte_velocidade = corte_velocidade
        self._filtro_valor = FiltroPassaBaixa()
        self._filtro_velocidade = FiltroPassaBaixa()
        self._valor_anterior: float | None = None

    @staticmethod
    def _alpha(corte: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(corte, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def aplicar(self, valor: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-3

        if self._valor_anterior is None:
            velocidade = 0.0
        else:
            velocidade = (valor - self._valor_anterior) / dt
        self._valor_anterior = valor

        velocidade_suave = self._filtro_velocidade.aplicar(
            velocidade, self._alpha(self.corte_velocidade, dt)
        )
        corte = self.corte_minimo + self.beta * abs(velocidade_suave)
        return self._filtro_valor.aplicar(valor, self._alpha(corte, dt))


class SuavizadorDeOlhar:
    """Junta mediana movel + One Euro + zona morta para transformar o olhar cru em cursor estavel."""

    def __init__(self, config: Configuracoes, x_inicial: float, y_inicial: float) -> None:
        self.zona_morta_px = config.zona_morta_px
        self._amostras_x: deque[float] = deque(maxlen=config.janela_mediana)
        self._amostras_y: deque[float] = deque(maxlen=config.janela_mediana)
        self._filtro_x = FiltroUmEuro(config.corte_minimo, config.beta_velocidade, config.corte_velocidade)
        self._filtro_y = FiltroUmEuro(config.corte_minimo, config.beta_velocidade, config.corte_velocidade)
        self._x_publicado = x_inicial
        self._y_publicado = y_inicial
        self._ultimo_instante: float | None = None

    def esquecer_historico(self) -> None:
        """Chamado quando o rastreamento se perde: evita um salto gigante ao voltar."""
        self._amostras_x.clear()
        self._amostras_y.clear()
        self._ultimo_instante = None

    def atualizar(self, alvo_x: float, alvo_y: float) -> tuple[float, float, bool]:
        agora = time.monotonic()
        dt = 1 / 30 if self._ultimo_instante is None else agora - self._ultimo_instante
        self._ultimo_instante = agora

        self._amostras_x.append(alvo_x)
        self._amostras_y.append(alvo_y)
        mediana_x = statistics.median(self._amostras_x)
        mediana_y = statistics.median(self._amostras_y)

        suave_x = self._filtro_x.aplicar(mediana_x, dt)
        suave_y = self._filtro_y.aplicar(mediana_y, dt)

        distancia = math.hypot(suave_x - self._x_publicado, suave_y - self._y_publicado)
        if distancia < self.zona_morta_px:
            return self._x_publicado, self._y_publicado, False

        self._x_publicado = suave_x
        self._y_publicado = suave_y
        return suave_x, suave_y, True


_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020


class SobreposicaoDeCursor:
    """Circulo vermelho, sempre no topo, que acompanha o cursor real na tela.

    Existe porque a posicao do olhar tem uma margem de erro esperada (principalmente no
    eixo vertical): em vez de fingir precisao de pixel, o circulo mostra visualmente essa
    folga, entao mirar "perto" do alvo dentro do circulo e o uso pretendido. A janela e
    transparente e clicavel-atraves (WS_EX_TRANSPARENT): ela nunca intercepta cliques do
    mouse, so desenha por cima de tudo.
    """

    _COR_CHAVE = "#010203"  # cor "impossivel" usada so como chave de transparencia.

    def __init__(self, raio: float) -> None:
        self.raio = raio
        self._diametro = max(int(raio * 2), 4)
        self._janela = tk.Tk()
        self._janela.overrideredirect(True)
        self._janela.attributes("-topmost", True)
        self._janela.attributes("-transparentcolor", self._COR_CHAVE)
        self._janela.geometry(f"{self._diametro}x{self._diametro}+0+0")

        self._canvas = tk.Canvas(
            self._janela,
            width=self._diametro,
            height=self._diametro,
            bg=self._COR_CHAVE,
            highlightthickness=0,
        )
        self._canvas.pack()
        margem = 2
        self._circulo = self._canvas.create_oval(
            margem, margem, self._diametro - margem, self._diametro - margem, outline="red", width=3
        )

        self._janela.update_idletasks()
        self._tornar_clicavel_atraves()

    def _tornar_clicavel_atraves(self) -> None:
        hwnd = self._janela.winfo_id()
        user32 = ctypes.windll.user32
        estilo_atual = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, estilo_atual | _WS_EX_LAYERED | _WS_EX_TRANSPARENT)

    def mover_para(self, x: float, y: float) -> None:
        esquerda = int(x - self.raio)
        topo = int(y - self.raio)
        self._janela.geometry(f"+{esquerda}+{topo}")
        self._janela.update_idletasks()
        self._janela.update()

    def fechar(self) -> None:
        self._janela.destroy()


def limitar(valor: float, minimo: float, maximo: float) -> float:
    return max(minimo, min(maximo, valor))


def mapear_intervalo(
    valor: float,
    origem_minima: float,
    origem_maxima: float,
    destino_minimo: float,
    destino_maximo: float,
) -> float:
    valor = limitar(valor, origem_minima, origem_maxima)
    tamanho_origem = origem_maxima - origem_minima
    tamanho_destino = destino_maximo - destino_minimo
    return destino_minimo + ((valor - origem_minima) / tamanho_origem) * tamanho_destino


def mapear_olhar_para_tela(ponto_olhar, largura_tela: int, altura_tela: int, area_de_controle):
    esquerda, direita, topo, base = area_de_controle
    x = mapear_intervalo(ponto_olhar[0], esquerda, direita, 0, largura_tela - 1)
    y = mapear_intervalo(ponto_olhar[1], topo, base, 0, altura_tela - 1)
    return x, y


def criar_gerador_de_timestamps():
    """Garante timestamps sempre crescentes para o detector, mesmo com frames repetidos no mesmo milissegundo."""
    estado = {"ultimo": -1}

    def proximo() -> int:
        agora = int(time.monotonic() * 1000)
        if agora <= estado["ultimo"]:
            agora = estado["ultimo"] + 1
        estado["ultimo"] = agora
        return agora

    return proximo


def obter_limites_da_tela(pyautogui):
    """Retorna os limites do monitor PRIMARIO (nao do desktop virtual com todos os monitores).

    A janela de calibracao (o ponto que o usuario olha) so aparece em um monitor por vez —
    normalmente o primario. Se essa funcao devolvesse o desktop virtual inteiro (todos os
    monitores somados, como faz o HandCursor original), a fracao 0..1 aprendida na
    calibracao ficaria mapeada para uma area bem maior que a tela onde a calibracao
    realmente apareceu, causando um erro de escala grande — sobretudo no eixo horizontal
    com dois monitores lado a lado. Por isso o controle do mouse fica restrito ao monitor
    primario: e o preco de calibrar contra uma janela que so pode estar em um lugar.
    """
    largura, altura = pyautogui.size()
    return 0, 0, largura, altura


def razao_iris_no_olho(pontos_rosto, canto_a: int, canto_b: int, topo: int, base: int, iris: int):
    """Calcula a posicao relativa da iris dentro do olho (0 a 1 nos dois eixos)."""
    ponto_a = pontos_rosto[canto_a]
    ponto_b = pontos_rosto[canto_b]
    ponto_topo = pontos_rosto[topo]
    ponto_base = pontos_rosto[base]
    ponto_iris = pontos_rosto[iris]

    x_minimo, x_maximo = sorted((ponto_a.x, ponto_b.x))
    y_minimo, y_maximo = sorted((ponto_topo.y, ponto_base.y))

    largura_olho = max(x_maximo - x_minimo, 1e-6)
    altura_olho = max(y_maximo - y_minimo, 1e-6)

    razao_x = (ponto_iris.x - x_minimo) / largura_olho
    razao_y = (ponto_iris.y - y_minimo) / altura_olho
    return razao_x, razao_y


def estimar_ponto_de_olhar(pontos_rosto):
    razao_x_direito, razao_y_direito = razao_iris_no_olho(
        pontos_rosto,
        OLHO_DIREITO_CANTO_EXTERNO,
        OLHO_DIREITO_CANTO_INTERNO,
        OLHO_DIREITO_TOPO,
        OLHO_DIREITO_BASE,
        IRIS_DIREITA_CENTRO,
    )
    razao_x_esquerdo, razao_y_esquerdo = razao_iris_no_olho(
        pontos_rosto,
        OLHO_ESQUERDO_CANTO_INTERNO,
        OLHO_ESQUERDO_CANTO_EXTERNO,
        OLHO_ESQUERDO_TOPO,
        OLHO_ESQUERDO_BASE,
        IRIS_ESQUERDA_CENTRO,
    )
    razao_x = (razao_x_direito + razao_x_esquerdo) / 2
    razao_y = (razao_y_direito + razao_y_esquerdo) / 2
    return razao_x, razao_y


def calcular_features_de_olhar(pontos_rosto):
    """Combina a posicao da iris com uma pista da pose da cabeca.

    A posicao da iris dentro do olho, por si so, muda muito quando a cabeca se move
    (nao so quando os olhos se movem). Incluir a posicao do nariz dentro do contorno
    do rosto deixa a regressao de calibracao aprender parte dessa compensacao.
    """
    razao_x, razao_y = estimar_ponto_de_olhar(pontos_rosto)

    nariz = pontos_rosto[NARIZ_PONTA]
    tempora_esquerda = pontos_rosto[ROSTO_TEMPORA_ESQUERDA]
    tempora_direita = pontos_rosto[ROSTO_TEMPORA_DIREITA]
    testa = pontos_rosto[ROSTO_TESTA]
    queixo = pontos_rosto[ROSTO_QUEIXO]

    largura_rosto = max(tempora_direita.x - tempora_esquerda.x, 1e-6)
    altura_rosto = max(queixo.y - testa.y, 1e-6)

    pose_x = (nariz.x - tempora_esquerda.x) / largura_rosto
    pose_y = (nariz.y - testa.y) / altura_rosto

    return razao_x, razao_y, pose_x, pose_y


def expandir_features_de_olhar(features_brutas):
    """Acrescenta interacao e termos quadraticos de razao_x/razao_y.

    Diagnostico real mostrou que olhar mais para os lados tambem distorce a leitura
    vertical da iris (efeito geometrico de uma unica camera 2D, nao ruido aleatorio):
    a mesma altura de olhar dava razao_y sistematicamente menor quando razao_x era maior.
    Um modelo puramente linear em razao_x e razao_y nao capta esse acoplamento; o termo
    de interacao (razao_x * razao_y) e os quadraticos dao a regressao capacidade de
    aprender essa distorcao a partir dos proprios pontos de calibracao.
    """
    razao_x, razao_y, pose_x, pose_y = features_brutas
    return (
        razao_x,
        razao_y,
        razao_x * razao_y,
        razao_x * razao_x,
        razao_y * razao_y,
        pose_x,
        pose_y,
    )


def obter_pontuacao_blendshape(blendshapes, nome: str) -> float:
    for categoria in blendshapes:
        if categoria.category_name == nome:
            return categoria.score
    return 0.0


def carregar_dependencias():
    print("Carregando OpenCV...", flush=True)
    import cv2
    import numpy as np

    print("Carregando MediaPipe... pode demorar alguns segundos na primeira vez.", flush=True)
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        FaceLandmarksConnections,
        RunningMode,
    )

    print("Carregando controle do mouse...", flush=True)
    import pyautogui

    return (
        cv2,
        np,
        mp,
        BaseOptions,
        FaceLandmarker,
        FaceLandmarkerOptions,
        FaceLandmarksConnections,
        RunningMode,
        pyautogui,
    )


def desenhar_area_de_controle(cv2, quadro, area_de_controle) -> None:
    esquerda, direita, topo, base = area_de_controle
    altura, largura, _ = quadro.shape
    canto_superior_esquerdo = (int(esquerda * largura), int(topo * altura))
    canto_inferior_direito = (int(direita * largura), int(base * altura))
    cv2.rectangle(quadro, canto_superior_esquerdo, canto_inferior_direito, (255, 255, 255), 1)


def redimensionar_previa(cv2, quadro, largura_previa: int):
    altura, largura, _ = quadro.shape
    altura_previa = int(altura * (largura_previa / largura))
    return cv2.resize(quadro, (largura_previa, altura_previa))


def criar_detector_de_rosto(
    BaseOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
    config: Configuracoes,
):
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "Modelo do rosto nao encontrado. Rode: "
            ".\\.venv\\Scripts\\python.exe download_model.py"
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(CAMINHO_MODELO)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=config.confianca_deteccao,
        min_face_presence_confidence=config.confianca_deteccao,
        min_tracking_confidence=config.confianca_rastreamento,
    )
    return FaceLandmarker.create_from_options(options)


def desenhar_olhos(cv2, quadro, pontos_rosto) -> None:
    altura, largura, _ = quadro.shape
    indices_dos_olhos = (
        OLHO_DIREITO_CANTO_EXTERNO,
        OLHO_DIREITO_CANTO_INTERNO,
        OLHO_DIREITO_TOPO,
        OLHO_DIREITO_BASE,
        IRIS_DIREITA_CENTRO,
        OLHO_ESQUERDO_CANTO_INTERNO,
        OLHO_ESQUERDO_CANTO_EXTERNO,
        OLHO_ESQUERDO_TOPO,
        OLHO_ESQUERDO_BASE,
        IRIS_ESQUERDA_CENTRO,
    )
    for indice in indices_dos_olhos:
        ponto = pontos_rosto[indice]
        centro = (int(ponto.x * largura), int(ponto.y * altura))
        cv2.circle(quadro, centro, 3, (255, 210, 60), -1)

    for indice_iris in (IRIS_DIREITA_CENTRO, IRIS_ESQUERDA_CENTRO):
        ponto = pontos_rosto[indice_iris]
        centro = (int(ponto.x * largura), int(ponto.y * altura))
        cv2.circle(quadro, centro, 5, (35, 190, 95), 2)


def desenhar_diagnostico_ao_vivo(cv2, quadro, pontos_rosto, alvo_x, alvo_y, largura_tela, altura_tela) -> None:
    """Mostra os numeros crus do rastreamento na propria janela da camera.

    Serve para diagnosticar de onde vem o erro: se razao_x/razao_y quase nao mudam
    quando voce move os olhos, o problema e a leitura do olhar em si (camera, distancia,
    iluminacao), nao a calibracao ou a suavizacao.
    """
    razao_x, razao_y, pose_x, pose_y = calcular_features_de_olhar(pontos_rosto)
    rel_x = alvo_x / max(largura_tela - 1, 1)
    rel_y = alvo_y / max(altura_tela - 1, 1)
    linhas = (
        f"razao_olho x={razao_x:.3f} y={razao_y:.3f}   pose_cabeca x={pose_x:.3f} y={pose_y:.3f}",
        f"alvo na tela (fracao) x={rel_x:.3f} y={rel_y:.3f}",
    )
    altura_quadro = quadro.shape[0]
    for indice, linha in enumerate(linhas):
        cv2.putText(
            quadro,
            linha,
            (20, altura_quadro - 45 + indice * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def capturar_quadro_processado(cv2, mp, camera, detector_rosto, proximo_tempo_ms):
    """Le um frame da camera, espelha e roda o detector. Retorna (resultado, quadro) ou (None, None)."""
    camera_ok, quadro = camera.read()
    if not camera_ok:
        return None, None

    quadro = cv2.flip(quadro, 1)
    rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
    imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector_rosto.detect_for_video(imagem, proximo_tempo_ms())
    return resultado, quadro


_XS_CALIBRACAO = (0.06, 0.37, 0.63, 0.94)
_YS_CALIBRACAO = (0.08, 0.36, 0.64, 0.92)
PONTOS_DE_CALIBRACAO = tuple((x, y) for y in _YS_CALIBRACAO for x in _XS_CALIBRACAO)
JANELA_CALIBRACAO = "Calibracao do olhar"
AMOSTRAS_MINIMAS_POR_PONTO = 15
AMOSTRAS_MAXIMAS_POR_PONTO = 30
DESVIO_MAXIMO_ESTAVEL = 0.012


def _esta_estavel(leituras) -> bool:
    """So aceita o ponto quando a leitura parou de se mover, nao so quando ja tem amostras.

    Sem isso, os primeiros frames de cada ponto (enquanto o olho ainda esta migrando
    para o alvo) entram na media/mediana e distorcem a calibracao inteira.
    """
    if len(leituras) < AMOSTRAS_MINIMAS_POR_PONTO:
        return False
    recentes = leituras[-AMOSTRAS_MINIMAS_POR_PONTO:]
    razao_x = [amostra[0] for amostra in recentes]
    razao_y = [amostra[1] for amostra in recentes]
    return statistics.pstdev(razao_x) < DESVIO_MAXIMO_ESTAVEL and statistics.pstdev(razao_y) < DESVIO_MAXIMO_ESTAVEL


def calibrar_regressao_de_olhar(cv2, np, mp, camera, detector_rosto, proximo_tempo_ms):
    """Pede para o usuario olhar para uma grade 4x4 de pontos na tela.

    Cada ponto guarda (posicao da iris + pose da cabeca) -> posicao real na tela.
    Com 16 pontos ajustamos por minimos quadrados uma regressao linear para X e para Y,
    o que compensa parte do erro causado por pequenos movimentos de cabeca — algo que
    uma simples area de controle fixa nao consegue fazer.

    So aceita ESPACO quando a leitura do olhar ja esta estavel (o texto muda para
    "pronto"): isso evita capturar o momento em que o olho ainda esta se movendo
    para o ponto, que e uma das maiores fontes de erro na calibracao.
    Pressione 's' para pular e usar a area de controle fixa padrao.
    """
    largura_janela, altura_janela = 1280, 720
    cv2.namedWindow(JANELA_CALIBRACAO, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(JANELA_CALIBRACAO, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        cv2.resizeWindow(JANELA_CALIBRACAO, largura_janela, altura_janela)

    linhas_x = []
    alvos_x = []
    alvos_y = []

    for indice_ponto, (rel_x, rel_y) in enumerate(PONTOS_DE_CALIBRACAO):
        leituras = []
        while True:
            resultado, quadro = capturar_quadro_processado(cv2, mp, camera, detector_rosto, proximo_tempo_ms)
            if quadro is None:
                continue

            if resultado.face_landmarks:
                leituras.append(calcular_features_de_olhar(resultado.face_landmarks[0]))
                if len(leituras) > AMOSTRAS_MAXIMAS_POR_PONTO:
                    leituras.pop(0)

            pronto = _esta_estavel(leituras)

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
            situacao = "pronto" if pronto else f"estabilizando... ({len(leituras)} leituras)"
            cv2.putText(
                canvas,
                situacao,
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
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
                linhas_x.append((*expandir_features_de_olhar(mediana), 1.0))
                alvos_x.append(rel_x)
                alvos_y.append(rel_y)
                break

    cv2.destroyWindow(JANELA_CALIBRACAO)
    pesos_x, pesos_y = ajustar_regressao_de_olhar(np, linhas_x, alvos_x, alvos_y)
    imprimir_diagnostico_de_calibracao(np, linhas_x, alvos_x, alvos_y, pesos_x, pesos_y)
    return pesos_x, pesos_y


def imprimir_diagnostico_de_calibracao(np, linhas, alvos_x, alvos_y, pesos_x, pesos_y) -> None:
    """Mostra no console o quao bem a regressao explica os proprios pontos de calibracao.

    Se o erro aqui ja for grande, o problema esta na leitura do olhar (ruido, faixa de
    variacao pequena, deteccao instavel) e nao na regressao em si. Se o erro aqui for
    pequeno mas o cursor ainda erra muito no uso real, o problema e mais provavelmente
    o sinal mudando entre a calibracao e o uso (cabeca em posicao diferente, luz diferente).
    """
    print("\n--- Diagnostico da calibracao ---")
    print("razao_x  razao_y  pose_x  pose_y  |  alvo_x  alvo_y  |  previsto_x  previsto_y  |  erro_x  erro_y")
    erros_x = []
    erros_y = []
    for linha, alvo_x, alvo_y in zip(linhas, alvos_x, alvos_y):
        previsto_x = sum(f * p for f, p in zip(linha, pesos_x))
        previsto_y = sum(f * p for f, p in zip(linha, pesos_y))
        erro_x = previsto_x - alvo_x
        erro_y = previsto_y - alvo_y
        erros_x.append(erro_x)
        erros_y.append(erro_y)
        razao_x, razao_y, _razao_xy, _razao_x2, _razao_y2, pose_x, pose_y, _bias = linha
        print(
            f"{razao_x:.3f}    {razao_y:.3f}    {pose_x:.3f}   {pose_y:.3f}   |  "
            f"{alvo_x:.3f}   {alvo_y:.3f}   |  {previsto_x:+.3f}      {previsto_y:+.3f}      |  "
            f"{erro_x:+.3f}   {erro_y:+.3f}"
        )

    faixa_razao_x = max(l[0] for l in linhas) - min(l[0] for l in linhas)
    faixa_razao_y = max(l[1] for l in linhas) - min(l[1] for l in linhas)
    print(f"\nFaixa observada de razao_x entre os pontos de calibracao: {faixa_razao_x:.3f}")
    print(f"Faixa observada de razao_y entre os pontos de calibracao: {faixa_razao_y:.3f}")
    print("(se essa faixa for pequena, ex: menor que 0.05, o sinal da iris quase nao varia")
    print(" com o olhar nessa camera/distancia, e nenhuma regressao vai resolver sozinha)")
    print(f"Erro medio absoluto no proprio treino: x={np.mean(np.abs(erros_x)):.3f}  y={np.mean(np.abs(erros_y)):.3f}")
    nomes_pesos = "razao_x, razao_y, razao_x*razao_y, razao_x^2, razao_y^2, pose_x, pose_y, bias"
    print(f"pesos_x ({nomes_pesos}): {[round(float(p), 3) for p in pesos_x]}")
    print(f"pesos_y ({nomes_pesos}): {[round(float(p), 3) for p in pesos_y]}")
    print("(se o peso de razao_y em pesos_y estiver em 0.0, o eixo vertical do cursor")
    print(" vira uma constante — e e exatamente isso que trava o cursor no topo/base)")
    print("--- Fim do diagnostico ---\n")


_INDICES_SEMPRE_MANTIDOS = (0, 1)  # razao_x, razao_y: nunca descartar, mesmo com pouca variancia.


def ajustar_regressao_de_olhar(np, linhas, alvos_x, alvos_y, regularizacao: float = 1e-3):
    """Ajusta pesos_x e pesos_y por minimos quadrados, com dois cuidados importantes:

    1. Descarta colunas quase constantes (variancia muito baixa) antes de resolver — exceto
       razao_x e razao_y (indices 0 e 1, ver _INDICES_SEMPRE_MANTIDOS), que nunca sao
       descartadas. A remocao evita, por exemplo, que a coluna de pose da cabeca quase
       constante (usuario mantendo a cabeca parada, o correto a se fazer) deixe o sistema
       quase singular e devolva coeficientes gigantes que amplificam ruido. Mas se razao_x
       ou razao_y forem descartadas por esse mesmo motivo, o eixo correspondente do cursor
       vira uma constante e trava — foi exatamente esse bug que travava o cursor no topo
       da tela, so respondendo no eixo horizontal.
    2. Usa uma pequena regularizacao (ridge) para manter os coeficientes das colunas
       restantes limitados mesmo com poucas amostras de calibracao.
    """
    matriz_completa = np.array(linhas, dtype=float)
    coluna_bias = matriz_completa.shape[1] - 1

    variancias = matriz_completa[:, :coluna_bias].var(axis=0)
    indices_uteis = [
        i
        for i, variancia in enumerate(variancias)
        if variancia > 1e-5 or i in _INDICES_SEMPRE_MANTIDOS
    ]
    indices_uteis.append(coluna_bias)

    matriz_reduzida = matriz_completa[:, indices_uteis]
    identidade = np.eye(matriz_reduzida.shape[1])
    gramiana = matriz_reduzida.T @ matriz_reduzida + regularizacao * identidade

    pesos_x = np.zeros(matriz_completa.shape[1])
    pesos_y = np.zeros(matriz_completa.shape[1])
    pesos_reduzidos_x = np.linalg.solve(gramiana, matriz_reduzida.T @ np.array(alvos_x, dtype=float))
    pesos_reduzidos_y = np.linalg.solve(gramiana, matriz_reduzida.T @ np.array(alvos_y, dtype=float))
    for indice_reduzido, indice_completo in enumerate(indices_uteis):
        pesos_x[indice_completo] = pesos_reduzidos_x[indice_reduzido]
        pesos_y[indice_completo] = pesos_reduzidos_y[indice_reduzido]

    return pesos_x, pesos_y


def criar_calculador_de_alvo_por_regressao(pesos_x, pesos_y, tela_esquerda, tela_topo, largura_tela, altura_tela):
    def calcular(pontos_rosto):
        features = expandir_features_de_olhar(calcular_features_de_olhar(pontos_rosto)) + (1.0,)
        rel_x = limitar(sum(f * p for f, p in zip(features, pesos_x)), 0.0, 1.0)
        rel_y = limitar(sum(f * p for f, p in zip(features, pesos_y)), 0.0, 1.0)
        x = tela_esquerda + rel_x * (largura_tela - 1)
        y = tela_topo + rel_y * (altura_tela - 1)
        return x, y

    return calcular


def criar_calculador_de_alvo_por_area(area_de_controle, tela_esquerda, tela_topo, largura_tela, altura_tela):
    def calcular(pontos_rosto):
        razao_x, razao_y = estimar_ponto_de_olhar(pontos_rosto)
        x, y = mapear_olhar_para_tela((razao_x, razao_y), largura_tela, altura_tela, area_de_controle)
        return x + tela_esquerda, y + tela_topo

    return calcular


def principal() -> None:
    (
        cv2,
        np,
        mp,
        BaseOptions,
        FaceLandmarker,
        FaceLandmarkerOptions,
        FaceLandmarksConnections,
        RunningMode,
        pyautogui,
    ) = carregar_dependencias()

    config = Configuracoes()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0
    tela_esquerda, tela_topo, largura_tela, altura_tela = obter_limites_da_tela(pyautogui)

    camera = cv2.VideoCapture(config.indice_camera)
    if not camera.isOpened():
        raise RuntimeError("Nao foi possivel abrir a camera.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.largura_camera)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.altura_camera)
    camera.set(cv2.CAP_PROP_FPS, config.fps_camera)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITULO_JANELA, config.largura_previa, 720)
    try:
        cv2.setWindowProperty(TITULO_JANELA, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    detector_rosto = criar_detector_de_rosto(
        BaseOptions,
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
        config,
    )
    proximo_tempo_ms = criar_gerador_de_timestamps()

    area_de_controle = (
        config.controle_esquerda,
        config.controle_direita,
        config.controle_topo,
        config.controle_base,
    )
    resultado_calibracao = calibrar_regressao_de_olhar(cv2, np, mp, camera, detector_rosto, proximo_tempo_ms)
    if resultado_calibracao is not None:
        pesos_x, pesos_y = resultado_calibracao
        calcular_alvo = criar_calculador_de_alvo_por_regressao(
            pesos_x, pesos_y, tela_esquerda, tela_topo, largura_tela, altura_tela
        )
    else:
        calcular_alvo = criar_calculador_de_alvo_por_area(
            area_de_controle, tela_esquerda, tela_topo, largura_tela, altura_tela
        )

    mouse_x = tela_esquerda + (largura_tela / 2)
    mouse_y = tela_topo + (altura_tela / 2)
    suavizador = SuavizadorDeOlhar(config, mouse_x, mouse_y)
    sobreposicao = SobreposicaoDeCursor(config.raio_do_cursor)
    olhos_estavam_fechados = False
    inicio_piscada: float | None = None
    mouse_esta_pressionado = False

    try:
        while True:
            resultado, quadro = capturar_quadro_processado(cv2, mp, camera, detector_rosto, proximo_tempo_ms)
            if quadro is None:
                break

            if resultado.face_landmarks:
                pontos_rosto = resultado.face_landmarks[0]
                blendshapes = resultado.face_blendshapes[0] if resultado.face_blendshapes else []

                piscar_esquerdo = obter_pontuacao_blendshape(blendshapes, "eyeBlinkLeft")
                piscar_direito = obter_pontuacao_blendshape(blendshapes, "eyeBlinkRight")
                sobrancelha = obter_pontuacao_blendshape(blendshapes, "browInnerUp")

                olhos_fechados = (
                    piscar_esquerdo > config.limite_piscar and piscar_direito > config.limite_piscar
                )
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

                    alvo_x, alvo_y = calcular_alvo(pontos_rosto)
                    mouse_x, mouse_y, deve_mover = suavizador.atualizar(alvo_x, alvo_y)
                    if deve_mover:
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0)

                    desenhar_diagnostico_ao_vivo(cv2, quadro, pontos_rosto, alvo_x, alvo_y, largura_tela, altura_tela)

                desenhar_olhos(cv2, quadro, pontos_rosto)

                if olhos_fechados:
                    cv2.putText(
                        quadro,
                        "Olhos fechados: solte rapido para clique esquerdo, devagar para direito",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (80, 80, 255),
                        2,
                    )
                elif sobrancelha_levantada:
                    cv2.putText(
                        quadro,
                        "Sobrancelha levantada: arrastando",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (35, 190, 95),
                        2,
                    )
            else:
                suavizador.esquecer_historico()
                if mouse_esta_pressionado:
                    pyautogui.mouseUp(button="left")
                    mouse_esta_pressionado = False
                    olhos_estavam_fechados = False
                    inicio_piscada = None

            if resultado_calibracao is None:
                desenhar_area_de_controle(cv2, quadro, area_de_controle)
            previa = redimensionar_previa(cv2, quadro, config.largura_previa)
            cv2.imshow(TITULO_JANELA, previa)

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
    principal()
