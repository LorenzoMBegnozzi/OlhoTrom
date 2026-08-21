# EyeCursor

Controle o cursor do mouse com o olhar usando a camera.

## Como funciona

- A posicao da iris dentro dos olhos move o cursor (media dos dois olhos).
- O controle fica restrito ao **monitor primario** (onde a tela de calibracao aparece). Veja "Multiplos monitores" abaixo.
- Um **circulo vermelho** acompanha o cursor pela tela (por cima de qualquer janela, sem bloquear cliques). Ele marca a margem de erro esperada do olhar — mire perto do alvo dentro do circulo, nao no pixel exato. O tamanho vem de `raio_do_cursor` em `Configuracoes`.
- Feche os dois olhos rapidamente e abra de novo: clique esquerdo.
- Feche os dois olhos por mais tempo e abra de novo: clique direito.
- Levante as sobrancelhas e mantenha: pressiona o botao esquerdo para arrastar ou selecionar, continuando a mover o cursor com o olhar. Abaixe as sobrancelhas para soltar.
- A janela da camera mostra os pontos dos olhos e da iris.
- O retangulo branco na camera e a area de controle: olhar dentro dele alcanca a tela inteira.
- Pressione `q` na janela da camera para sair.

## Calibracao ao iniciar

Antes da janela da camera, o programa abre uma tela de calibracao com uma grade de 16 pontos (4x4). Para cada ponto: olhe fixo para ele e espere o ponto ficar **verde** e o texto mudar para "pronto" (o programa exige que a leitura do olhar realmente pare de variar antes de aceitar — nao basta esperar um numero fixo de frames). So depois disso pressione **ESPACO**. Pressione `s` em qualquer ponto para pular e usar a area de controle fixa padrao de `Configuracoes` (bem menos precisa).

Com os 16 pontos, o programa ajusta por minimos quadrados uma regressao que combina a posicao da iris (incluindo um termo de interacao entre os eixos, que compensa uma distorcao real: olhar mais para o lado tambem muda um pouco a leitura vertical numa camera 2D comum) com a posicao do nariz dentro do contorno do rosto — essa segunda parte compensa parcialmente o desvio causado quando a cabeca se move um pouco.

## Multiplos monitores

O controle do cursor fica restrito ao monitor primario do Windows (o mesmo onde a tela de calibracao aparece em tela cheia). Isso e proposital: a calibracao so pode acontecer contra a tela onde a janela de calibracao de fato apareceu, entao mapear o resultado para o desktop virtual inteiro (todos os monitores somados) faria o cursor sair muito do lugar — por exemplo, olhar para o canto direito do monitor calibrado seria interpretado como "meio caminho" se houver um segundo monitor do mesmo tamanho ao lado. Se voce usa dois monitores, confirme qual e o monitor primario do Windows (Configuracoes > Sistema > Tela) antes de calibrar, e vai usar o mouse so nele.

Dicas para uma boa calibracao:

- Mantenha a cabeca parada (mesma distancia e angulo) durante toda a calibracao e o uso depois. Regressao ajuda, mas nao substitui manter a cabeca estavel.
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
- `raio_do_cursor`: raio em pixels do circulo vermelho que acompanha o cursor. Aumente se estiver dificil de "acertar" o alvo; diminua se quiser mais precisao visual.
- `largura_previa`: largura da janela de video para gravacao.
- `indice_camera`: troque para `1` se a camera principal nao abrir.

O PyAutoGUI tem um failsafe: mover o mouse rapidamente para o canto superior esquerdo da tela interrompe a automacao.

## Limitacoes

Rastreamento de olhar por webcam comum (sem infravermelho, sem estimativa de profundidade) e inerentemente menos preciso que rastreamento por mao: pequenos movimentos de cabeca, pouca luz, oculos com reflexo ou camera de baixa resolucao pioram a leitura. A calibracao por regressao reduz bastante o erro, mas nao elimina a sensibilidade a movimento de cabeca. Se o cursor tremer muito depois de calibrado, ajuste `corte_minimo`/`beta_velocidade`/`zona_morta_px`; se ele desviar de onde voce olha, confira se voce manteve a cabeca na mesma posicao da calibracao e recalibre.
