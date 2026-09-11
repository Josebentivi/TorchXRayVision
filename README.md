# Aula de IA em Radiologia

Material didático interativo para ensinar **estudantes de medicina** a usar, ler e
desconfiar de sistemas de inteligência artificial aplicados à radiografia de tórax.

O objetivo não é formar programadores. É que o aluno saia capaz de operar um modelo
de verdade, entender o que o número na tela significa e fazer as perguntas certas
quando um sistema desses for apresentado ao serviço dele.

> ⚠️ **Este programa é material didático.** Não é dispositivo médico, não tem registro
> na ANVISA, no FDA ou na CE, e **não pode ser usado para tomar, apoiar ou justificar
> qualquer decisão clínica sobre qualquer pessoa real.** Não envie imagens de pacientes
> identificáveis.

> ℹ️ Este repositório **não é** a biblioteca TorchXRayVision. É um material de aula que
> a utiliza. A biblioteca original está em [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision).

---

## O que a aula faz

Tudo roda em um único arquivo Python, com interface web. O aluno escolhe uma
radiografia — as de exemplo baixam sozinhas — e percorre oito módulos que rodam
modelos reais, os mesmos pesos publicados por grupos do NIH, de Stanford, do MIT,
do Hospital San Juan e da RSNA.

Algumas coisas que ele vai ver acontecer na tela:

- Uma radiografia de 2.000 × 2.500 pixels sendo reduzida a 224 × 224 — **sobra cerca de 1%**
  da informação espacial antes de a rede olhar para ela.
- Que `0,62` na saída do modelo **não quer dizer** 62% de chance de ter a doença.
- O mesmo modelo, com 90% de sensibilidade e 90% de especificidade, produzindo
  **metade dos alarmes falsa** numa prevalência de 10%.
- A mesma imagem recebendo respostas diferentes de modelos treinados em hospitais
  diferentes — inclusive dois treinados nas **mesmas imagens**, mudando só o programa
  que leu os laudos para gerar os rótulos.
- A saída mudar ao girar, espelhar ou escurecer a imagem, sem que nada clínico tenha mudado.
- Uma rede estimando **idade** e **etnia autodeclarada** a partir do raio X, o que demonstra
  que a imagem carrega sinais não clínicos recuperáveis — e, portanto, disponíveis como
  atalho para qualquer outro modelo treinado nessas bases.

---

## Como está organizado

Cada módulo é precedido por uma página de **Preparação**, que ensina o vocabulário
necessário antes de o módulo começar. Elas existem porque o público é de medicina e
não tem por que já saber o que é um logito, um rótulo ou uma máscara. Todas funcionam
sem radiografia carregada, para leitura prévia, e terminam com perguntas de
autoverificação.

| # | Preparação ensina | Módulo demonstra |
|---|---|---|
| 1 | Pixel, profundidade de bits, DICOM, janelamento, PA × AP | O que o pré-processamento faz com a imagem |
| 2 | Rede neural, pesos, logito, sigmoide, ponto de corte | Como ler as 18 saídas do modelo |
| 3 | Sensibilidade, VPP, prevalência, razão de verossimilhança | Por que um modelo excelente pode ser inútil no seu serviço |
| 4 | Convolução, mapa de características, explicabilidade | Onde o modelo olhou, com mapa de calor exato |
| 5 | Bases de dados, rotulação automática, mudança de distribuição | Modelos treinados em populações diferentes discordando |
| 6 | Robustez, invariância, gama, lateralidade | Girar, espelhar e escurecer mudando o "laudo" |
| 7 | Correlação × causa, atalho, viés algorítmico, equidade | Idade e etnia autodeclarada extraídas do raio X |
| 8 | Classificação × detecção × segmentação, Dice, ICT | Contornos de 14 estruturas e o índice cardiotorácico |

Antes de tudo há uma página de abertura com a apresentação do autor, e uma página
**Início** com o roteiro completo.

---

## Instalação

Precisa de **Python 3.10 ou mais novo** — que é o mínimo exigido pelas versões atuais
do PyTorch. Roda em CPU: não precisa de placa de vídeo.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -m pip install torchxrayvision pydicom streamlit scikit-image matplotlib pandas
```

> **Se você já tem PyTorch instalado**, não rode o primeiro comando sem antes conferir a
> versão. O `pip` resolve `torchvision` para a versão mais recente do índice e arrasta
> junto um PyTorch novo — o que pode significar baixar mais de 100 MB à toa. Confira com
> `python -c "import torch; print(torch.__version__)"` e fixe o `torchvision` par
> (0.27 ↔ 2.12, 0.28 ↔ 2.13, 0.29 ↔ 2.14). Um `--dry-run` mostra o que o pip pretende fazer.

---

## Como rodar

### No seu computador

```bash
streamlit run aula_ia_radiologia.py
```

O navegador abre sozinho em `http://localhost:8501`. Para encerrar, `Ctrl+C` no terminal.

### No Google Colab

O Streamlit não aparece direto no notebook: precisa de um túnel. Rode
`python aula_ia_radiologia.py` que o próprio programa imprime as células prontas,
com `localtunnel` e como obter a senha do túnel.

### Primeiro uso

Os pesos dos modelos são baixados sob demanda, só quando você abre o módulo que
usa cada um, e ficam em cache depois disso:

| Modelo | Tamanho | Usado no módulo |
|---|---|---|
| DenseNet-121 (cada um dos 7) | ~30 MB | 1 a 6 |
| Estimativa de idade (RIKEN) | ~440 MB | 7 |
| Experimento de Gichoya (Emory HITI) | ~85 MB | 7 |
| Segmentação PSPNet (ChestX-Det) | ~260 MB | 8 |

Se estiver planejando uma aula, vale abrir cada módulo uma vez antes, com uma
conexão boa. Se um download for interrompido, o arquivo fica pela metade no cache
e o programa avisa qual pasta apagar.

---

## Personalizar

Tudo o que costuma precisar de ajuste está reunido no **BLOCO 1** do arquivo, logo
no começo:

- `AUTOR`, `AUTOR_WHATSAPP_*`, `AUTOR_EMAIL` — dados de contato da página de abertura.
- `AUTOR_BIO` — apresentação pessoal. Vem vazia; enquanto estiver assim, a seção não
  aparece na página. Aceita markdown.
- `ARQUIVO_FOTO_AUTOR` — se existir um `foto_autor.jpg` na pasta, ele vira a foto da
  página de abertura. Sem o arquivo, o layout se ajusta sozinho.
- `ACHADOS` — tradução e significado clínico dos 18 achados.
- `MODELOS` — quais modelos aparecem no seletor e o que se diz sobre cada base.

O arquivo é um só, com blocos numerados e comentados. Para achar uma seção, busque
por `BLOCO 4` ou `PREPARAÇÃO 5`, por exemplo.

---

## Ambiente testado

| | |
|---|---|
| Sistema | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.12.1+cpu |
| torchvision | 0.27.1+cpu |
| torchxrayvision | 1.5.4 |
| Streamlit | 1.43.2 |
| pydicom | 3.0.2 |

---

## Créditos

Os modelos e a infraestrutura vêm da biblioteca aberta **TorchXRayVision**:

> Cohen, J. P. et al. *TorchXRayVision: A library of chest X-ray datasets and models.*
> MIDL 2022. https://github.com/mlmed/torchxrayvision

As bases de treino dos modelos são NIH ChestX-ray14, CheXpert (Stanford), MIMIC-CXR
(Beth Israel Deaconess), PadChest (Hospital San Juan, Alicante) e RSNA Pneumonia
Detection Challenge.

Os dois estudos discutidos no módulo 7:

- Gichoya, J. W. et al. *AI recognition of patient race in medical imaging: a modelling study.*
  The Lancet Digital Health, 2022.
- Seyyed-Kalantari, L. et al. *Underdiagnosis bias of artificial intelligence algorithms
  applied to chest radiographs in under-served patient populations.* Nature Medicine, 2021.

### Licenças

O núcleo do TorchXRayVision é **Apache 2.0**, mas a licença **varia por subpacote**: os
*baseline models* usados nos módulos 7 e 8 (idade, etnia e segmentação) têm termos
próprios, que precisam ser verificados individualmente antes de qualquer uso além do
educacional. As radiografias em `imagens_exemplo/` vêm do repositório de testes da
biblioteca.

Este material ainda não tem um arquivo de licença definido.

---

## Autor

**João José Bentivi**

- WhatsApp: [(11) 96994-2000](https://wa.me/5511969942000)
- E-mail: [josebentivi@gmail.com](mailto:josebentivi@gmail.com)

Dúvidas sobre o conteúdo, sugestões, correções ou interesse em usar este material em
aula são bem-vindos.
