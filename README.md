# EyeCursor

Controle o cursor do mouse com o olhar usando a camera.

## Como funciona

Controle **hibrido** (olhos + cabeca), porque a posicao vertical da iris dentro do olho, sozinha, nao carrega sinal suficiente para apontar a tela com essa camera (confirmado testando na pratica — veja "Por que hibrido?" abaixo):

- **Horizontal**: posicao da iris dentro dos olhos (media dos dois olhos) — mova so os olhos para os lados, sem precisar mexer a cabeca.
- **Vertical**: incline a cabeca para cima/baixo (como se olhasse por cima ou por baixo de um oculos). Nao e a posicao vertical do olho — e a cabeca inteira que inclina.
- O controle fica restrito ao **monitor primario** (onde a tela de calibracao aparece). Veja "Multiplos monitores" abaixo.
- Um **circulo vermelho** acompanha o cursor pela tela (por cima de qualquer janela, sem bloquear cliques). Ele marca a margem de erro esperada do olhar — mire perto do alvo dentro do circulo, nao no pixel exato. O tamanho vem de `raio_do_cursor` em `Configuracoes`.
- Feche os dois olhos rapidamente e abra de novo: clique esquerdo.
- Feche os dois olhos por mais tempo e abra de novo: clique direito.
- Levante as sobrancelhas e mantenha: pressiona o botao esquerdo para arrastar ou selecionar, continuando a mover o cursor com o olhar. Abaixe as sobrancelhas para soltar.
- A janela da camera mostra os pontos dos olhos e da iris.
- O retangulo branco na camera e a area de controle: olhar dentro dele alcanca a tela inteira.
- Pressione `q` na janela da camera para sair.

## Por que hibrido?

Testamos primeiro so com os olhos (como o HandCursor faz com a mao). O eixo horizontal funcionou bem. O vertical, nao — em testes reais de calibracao, a posicao vertical da iris ficava praticamente achatada entre olhar para o topo e para a base da tela (a faixa observada era de ~0.05, contra ~0.18 no eixo horizontal), sem relacao clara com o alvo. Tentamos compensar com regressao mais complexa (termos quadraticos, correcao 3D de pose da cabeca) e o resultado nao melhorou — o problema e a falta de sinal em si, nao a matematica em cima dele. A inclinacao da cabeca (pitch) e um movimento muito maior e mais voluntario, com sinal bem mais forte, por isso virou o controle vertical.

## Calibracao ao iniciar

Antes da janela da camera, o programa abre uma tela de calibracao com uma grade de 16 pontos (4x4). Para cada ponto: olhe e incline a cabeca ate mirar nele, espere o ponto ficar **verde** e o texto mudar para "pronto" (o programa exige que a leitura realmente pare de variar antes de aceitar — nao basta esperar um numero fixo de frames). So depois disso pressione **ESPACO**. Pressione `s` em qualquer ponto para pular e usar a area de controle fixa padrao de `Configuracoes` (so pela iris nos dois eixos — bem menos precisa, principalmente no vertical).

Com os 16 pontos, o programa ajusta por minimos quadrados uma regressao linear: (posicao da iris, inclinacao da cabeca) -> posicao na tela.

## Multiplos monitores

O controle do cursor fica restrito ao monitor primario do Windows (o mesmo onde a tela de calibracao aparece em tela cheia). Isso e proposital: a calibracao so pode acontecer contra a tela onde a janela de calibracao de fato apareceu, entao mapear o resultado para o desktop virtual inteiro (todos os monitores somados) faria o cursor sair muito do lugar — por exemplo, olhar para o canto direito do monitor calibrado seria interpretado como "meio caminho" se houver um segundo monitor do mesmo tamanho ao lado. Se voce usa dois monitores, confirme qual e o monitor primario do Windows (Configuracoes > Sistema > Tela) antes de calibrar, e vai usar o mouse so nele.

Dicas para uma boa calibracao:

- Mantenha a mesma distancia da camera durante toda a calibracao e o uso depois (a inclinacao da cabeca e relativa a essa distancia). Recalibre se mudar de lugar.
- So pressione ESPACO depois que o ponto ficar verde e o texto mudar para "pronto". Se ele nunca ficar verde, o rastreamento esta instavel (luz ruim, muito longe da camera, ou olhos meio fechados) — ajuste isso antes de continuar.
- Calibre com a mesma iluminacao e distancia da camera que voce vai usar depois. Recalibre se trocar de lugar, de luz ou de posicao da camera.
- Se o cursor continuar impreciso apos calibrar, refaca a calibracao com mais cuidado (espere estabilizar antes de apertar ESPACO), ou ajuste `limite_piscar`/`limite_sobrancelha` se cliques acidentais estiverem interferindo.

## Instalar

Crie um ambiente virtual e instale as dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_model.py
```

Se o `mediapipe` nao instalar no Python 3.13, instale o Python 3.11 ou 3.12 e recrie o ambiente virtual com essa versao. O MediaPipe costuma liberar suporte a novas versoes do Python depois das bibliotecas principais.

## Rodar

```powershell
python olho.py
```

Na primeira execucao, o MediaPipe pode levar alguns segundos para carregar. Espere aparecer a janela da camera.

## Ajustes uteis

No arquivo `olho.py`, altere os valores em `Configuracoes`:

- `limite_piscar`: pontuacao minima do blendshape de piscar (0 a 1) para considerar os olhos fechados. Aumente se estiver disparando cliques sem querer.
- `tempo_clique_curto`: duracao maxima, em segundos, de uma piscada para virar clique esquerdo. Acima disso, conta como clique direito.
- `limite_sobrancelha`: pontuacao minima do blendshape de sobrancelha levantada para iniciar o arrasto.
- `controle_esquerda`, `controle_direita`, `controle_topo`, `controle_base`: area de controle fixa usada apenas quando voce pula a calibracao (tecla `s`).
- `janela_mediana`: quantos frames entram na mediana movel antes da suavizacao. Mais frames = mais estavel, mais atraso.
- `corte_minimo`, `beta_velocidade`, `corte_velocidade`: parametros do One Euro Filter que suaviza o cursor. `corte_minimo` mais baixo deixa o cursor mais parado quando o olhar esta quieto; `beta_velocidade` mais alto reduz o atraso quando o olhar se move rapido, mas deixa mais tremor.
- `zona_morta_px`: raio em pixels dentro do qual pequenas variacoes do olhar sao ignoradas e o cursor fica parado.
- `tempo_para_mover_s`: tempo, em segundos, que o olhar precisa se manter fora da zona morta antes do cursor de fato se mover. Filtra microssacadas (o olho nunca fica 100% parado, mesmo "olhando fixo"). Maior = cursor mais parado, mais atraso para reagir; menor = mais responsivo, mais suscetivel a tremores.
- `raio_do_cursor`: raio em pixels do circulo vermelho que acompanha o cursor. Aumente se estiver dificil de "acertar" o alvo; diminua se quiser mais precisao visual.
- `largura_previa`: largura da janela de video para gravacao.
- `indice_camera`: troque para `1` se a camera principal nao abrir.

O PyAutoGUI tem um failsafe: mover o mouse rapidamente para o canto superior esquerdo da tela interrompe a automacao.

## Limitacoes

Rastreamento de olhar por webcam comum (sem infravermelho, sem estimativa de profundidade) e inerentemente menos preciso que rastreamento por mao, e o controle vertical aqui depende de inclinar a cabeca em vez do olhar em si — precisa de um pouco mais de esforco fisico que so mover os olhos. Pouca luz, oculos com reflexo ou camera de baixa resolucao tambem pioram a leitura horizontal. Se o cursor tremer muito depois de calibrado, ajuste `corte_minimo`/`beta_velocidade`/`zona_morta_px`; se ele desviar muito, confira se voce esta na mesma distancia da camera usada na calibracao e recalibre.

## Modo experimental: MobileGaze (rede neural leve, ONNX)

`olho_gaze_onnx.py` e uma versao alternativa que troca o calculo geometrico da iris por
uma rede neural leve (MobileOne S0, do repositorio [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation),
MIT license) que preve o angulo do olhar (yaw/pitch) direto da imagem do rosto, via ONNX
Runtime. Reaproveita piscar/sobrancelha/suavizacao/overlay de `olho.py`.

(Antes disso, tentamos o L2CS-Net com ResNet50 — mas o link oficial dos pesos no Google
Drive esta fora do ar, e o modelo seria bem mais pesado de qualquer forma. O MobileGaze
tem pesos hospedados no GitHub Releases do proprio autor, oficial e estavel, e e ~20x mais
leve — sem precisar do `torch`.)

Instalar (leve, so `onnxruntime`, sem torch):

```powershell
pip install -r requirements-gaze-onnx.txt
python download_gaze_model.py
```

Antes de calibrar (16 pontos, alguns minutos), meca a velocidade na sua CPU:

```powershell
python olho_gaze_onnx.py --benchmark
```

Isso mede quantos milissegundos por frame a rede neural leva. Acima de ~33ms por frame
(menos de 30fps) o cursor vai parecer com atraso; o benchmark avisa se esta lento demais —
mas por ser um modelo bem leve (~5MB), a expectativa e que rode rapido mesmo numa CPU comum.
So depois disso vale a pena rodar `python olho_gaze_onnx.py` de verdade e passar pela
calibracao de 16 pontos.

Diferencas em relacao a `olho.py`: sem area de controle fixa como alternativa (se pular a
calibracao, o modo simplesmente nao funciona — nao ha um mapeamento razoavel de angulo de
olhar para tela sem calibrar); sem o diagnostico ao vivo na tela (`desenhar_diagnostico_ao_vivo`).

A classe `GazeEstimationONNX` em `olho_gaze_onnx.py` foi adaptada do `onnx_inference.py`
original do MobileGaze (mesma logica de pre/pos-processamento), so trocando a deteccao de
rosto por reaproveitar o MediaPipe (que ja roda por causa do piscar/sobrancelha), evitando
correr dois detectores de rosto por frame.
