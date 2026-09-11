# -*- coding: utf-8 -*-
"""
================================================================================
  RADIOLOGIA + IA NA PRÁTICA — Aula interativa com TorchXRayVision
================================================================================

PARA QUEM É ESTE ARQUIVO
------------------------
Para estudantes de medicina. O objetivo NÃO é te ensinar a programar redes
neurais. O objetivo é te ensinar a USAR, LER e DESCONFIAR de um sistema de IA
real aplicado à radiografia de tórax — que é exatamente a competência esperada
de um médico que vai conviver com essas ferramentas no hospital.

Você vai rodar, na sua própria máquina, os mesmos modelos que aparecem em
artigos de radiologia. Não é simulação: os pesos são os publicados pelos grupos
que treinaram as redes em bases como NIH ChestX-ray14, CheXpert, MIMIC-CXR,
PadChest e RSNA.

O QUE VOCÊ VAI APRENDER (um módulo por item)
--------------------------------------------
  1. O modelo não vê o que você vê  → o que o pré-processamento faz com a imagem
  2. Ler a saída do modelo          → 18 achados, e por que "0,73" NÃO é
                                      "73% de chance de ter a doença"
  3. Limiar, sensibilidade e Bayes  → por que o MESMO modelo tem VPP diferente
                                      no pronto-socorro e no rastreamento
  4. Onde o modelo olhou            → mapa de ativação sobreposto à radiografia
  5. Modelos discordam entre si     → a mesma imagem em redes treinadas em
                                      populações diferentes
  6. Modelos são frágeis            → girar, espelhar ou escurecer muda o "laudo"
  7. Modelos aprendem o que não     → idade e etnia autodeclarada a partir do
     deveriam                         RX, e o que isso significa para viés
  8. IA não é só classificar        → segmentação anatômica de 14 estruturas

--------------------------------------------------------------------------------
  AVISO IMPORTANTE — LEIA ANTES DE USAR
--------------------------------------------------------------------------------
  Este programa é MATERIAL DIDÁTICO. Não é dispositivo médico, não tem
  registro em nenhuma agência reguladora (ANVISA, FDA, CE) e NÃO PODE ser
  usado para tomar, apoiar ou justificar qualquer decisão clínica sobre
  qualquer pessoa real.

  Não envie para este programa imagens de pacientes identificáveis. Use as
  radiografias de exemplo que ele baixa automaticamente, ou imagens
  públicas / devidamente anonimizadas.
--------------------------------------------------------------------------------

COMO RODAR
----------
  1) Instale as dependências (uma vez só):

       python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
       python -m pip install torchxrayvision pydicom streamlit scikit-image matplotlib pandas

  2) Rode o aplicativo:

       streamlit run aula_ia_radiologia.py

     O navegador abre sozinho em http://localhost:8501

  3) Está no Google Colab? Rode `python aula_ia_radiologia.py` que o próprio
     programa imprime as células prontas para o Colab.

OBSERVAÇÃO SOBRE O PRIMEIRO USO
-------------------------------
Na primeira vez que você abrir cada módulo, o programa baixa os pesos do modelo
correspondente (cerca de 30 MB cada). Depois disso fica em cache e abre na hora.
Só precisa de internet nesse primeiro download.

Biblioteca usada: TorchXRayVision — https://github.com/mlmed/torchxrayvision
Para citar em trabalhos: Cohen et al., "TorchXRayVision: A library of chest
X-ray datasets and models", MIDL 2022.
"""

# ==============================================================================
# BLOCO 0 — DEPENDÊNCIAS
# ==============================================================================
# Antes de importar qualquer coisa pesada, conferimos o que está faltando. Assim,
# se você esqueceu de instalar algo, recebe uma mensagem clara em vez de um
# traceback de 40 linhas.

import importlib.util
import math
import sys
import tempfile
import textwrap
import urllib.request
from pathlib import Path

# ------------------------------------------------------------------------------
# Correção obrigatória no Windows — não remova.
#
# O terminal do Windows ainda usa, por padrão, uma tabela de caracteres antiga
# (cp1252) que não dá conta de acentos nem dos blocos "█" que o TorchXRayVision
# imprime na barra de progresso enquanto baixa os pesos dos modelos.
#
# Sem estas linhas, o PRIMEIRO download de cada modelo derruba o aplicativo
# inteiro com UnicodeEncodeError — e o erro aparece na tela do aluno como um
# traceback gigante, sem nenhuma relação aparente com a causa real.
# ------------------------------------------------------------------------------
for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # stdout pode não existir (serviço em segundo plano) ou já estar correto.
        pass

# Mapa: nome do módulo em Python -> nome do pacote no pip (nem sempre são iguais,
# veja "skimage" vs "scikit-image").
_DEPENDENCIAS = {
    "streamlit": "streamlit",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "skimage": "scikit-image",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchxrayvision": "torchxrayvision",
}

_FALTANDO = [
    pacote
    for modulo, pacote in _DEPENDENCIAS.items()
    if importlib.util.find_spec(modulo) is None
]


def _texto_instalacao(faltando):
    """Monta a mensagem de instalação, já com os comandos prontos para copiar."""
    return textwrap.dedent(
        f"""
        Faltam dependências para rodar esta aula: {", ".join(faltando)}

        Instale assim (o primeiro comando usa o repositório de CPU do PyTorch,
        que é bem menor e não exige placa de vídeo):

            python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            python -m pip install torchxrayvision pydicom streamlit scikit-image matplotlib pandas

        Depois rode de novo:

            streamlit run aula_ia_radiologia.py
        """
    ).strip()


if _FALTANDO:
    # Se o próprio Streamlit estiver disponível, mostramos o aviso no navegador.
    # Se nem ele existir, sobra o terminal mesmo.
    if "streamlit" not in _FALTANDO:
        import streamlit as st

        st.set_page_config(page_title="Aula de IA em Radiologia")
        st.error(_texto_instalacao(_FALTANDO))
        st.stop()
    else:
        print(_texto_instalacao(_FALTANDO), file=sys.stderr)
        sys.exit(1)

# A partir daqui está tudo instalado e podemos importar de verdade.
import logging

import numpy as np
import pandas as pd
import streamlit as st
import torch

# O Streamlit reclama no terminal sempre que funções com cache são chamadas fora
# de um servidor ativo ("missing ScriptRunContext"). São avisos inofensivos e que
# só confundem o aluno, então baixamos o nível do registro. Erros de verdade
# continuam aparecendo.
#
# Não basta ajustar o registrador "streamlit": a biblioteca fixa o nível em cada
# sub-registrador separadamente, então eles não herdam nada do pai. Por isso
# usamos a função oficial dela e, por garantia, varremos os que já existem.
try:
    from streamlit import logger as _registro_streamlit

    _registro_streamlit.set_log_level("error")
except Exception:
    pass

for _nome_registro in ["streamlit", *list(logging.root.manager.loggerDict)]:
    if _nome_registro.startswith("streamlit"):
        logging.getLogger(_nome_registro).setLevel(logging.ERROR)
import skimage.io
import skimage.transform

import matplotlib

# "Agg" é um backend que desenha em memória, sem tentar abrir janela do sistema
# operacional. É o backend correto para servidor web (Streamlit) e para o Colab.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# O Streamlit pode estar no tema claro ou no escuro, dependendo da configuração
# do navegador de quem abrir a página. O matplotlib não sabe disso e desenha com
# as cores que estiverem valendo. Se deixássemos por conta do acaso, um usuário
# no tema escuro poderia acabar com texto preto sobre fundo preto.
#
# Fixamos as cores das figuras aqui, de uma vez: fundo claro e texto escuro,
# legível nos dois temas. É a mesma escolha de um gráfico impresso em artigo.
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "#1f1f1f",
        "axes.labelcolor": "#1f1f1f",
        "axes.edgecolor": "#9a9a9a",
        "xtick.color": "#3d3d3d",
        "ytick.color": "#3d3d3d",
        "font.size": 9.5,
    }
)

import torchxrayvision as xrv


# ==============================================================================
# BLOCO 1 — DICIONÁRIOS DIDÁTICOS
# ==============================================================================
# Estes dicionários são o "glossário" da aula. Eles traduzem o vocabulário da
# rede neural (que é todo em inglês, herdado dos rótulos das bases americanas)
# para o vocabulário clínico em português.

# ------------------------------------------------------------------------------
# Os 18 achados que o modelo tenta detectar.
#
# ATENÇÃO A UM PONTO CONCEITUAL: esta lista NÃO é uma lista de diagnósticos. São
# rótulos que, na maioria das bases, foram extraídos AUTOMATICAMENTE do texto do
# laudo por um programa de processamento de linguagem natural. Ou seja: o modelo
# não aprendeu "o que é uma pneumonia"; ele aprendeu "quais imagens costumam
# aparecer quando um radiologista escreveu a palavra pneumonia no laudo". Essa
# diferença explica boa parte dos erros que você vai ver nesta aula.
# ------------------------------------------------------------------------------
ACHADOS = {
    "Atelectasis": (
        "Atelectasia",
        "Colapso alveolar com perda de volume; desvio de estruturas para o lado acometido.",
    ),
    "Consolidation": (
        "Consolidação",
        "Preenchimento do espaço alveolar (exsudato, sangue, pus). Broncograma aéreo é o sinal clássico.",
    ),
    "Infiltration": (
        "Infiltrado",
        "Termo vago e hoje desencorajado, mas presente na base NIH. Herdamos a imprecisão do rótulo original.",
    ),
    "Pneumothorax": (
        "Pneumotórax",
        "Ar no espaço pleural. Achado de alta prioridade — é o tipo de coisa que a triagem por IA promete pegar.",
    ),
    "Edema": (
        "Edema pulmonar",
        "Congestão / edema intersticial ou alveolar, em geral de origem cardiogênica.",
    ),
    "Emphysema": (
        "Enfisema",
        "Destruição de septos alveolares, com hiperinsuflação e retificação diafragmática.",
    ),
    "Fibrosis": (
        "Fibrose",
        "Padrão reticular, perda volumétrica e, em fases avançadas, faveolamento.",
    ),
    "Effusion": (
        "Derrame pleural",
        "Líquido no espaço pleural; velamento do seio costofrênico, menisco.",
    ),
    "Pneumonia": (
        "Pneumonia",
        "Rótulo clínico-radiológico. Lembre que pneumonia é diagnóstico clínico, não radiológico isolado.",
    ),
    "Pleural_Thickening": (
        "Espessamento pleural",
        "Pleura espessada, frequentemente sequelar (empiema prévio, asbesto, hemotórax).",
    ),
    "Cardiomegaly": (
        "Cardiomegalia",
        "Índice cardiotorácico > 0,5 em incidência PA. Em AP o coração é magnificado — cuidado com falso positivo.",
    ),
    "Nodule": (
        "Nódulo",
        "Opacidade arredondada de até 3 cm.",
    ),
    "Mass": (
        "Massa",
        "Opacidade arredondada maior que 3 cm.",
    ),
    "Hernia": (
        "Hérnia",
        "Hérnia diafragmática / hiatal. Achado raríssimo nas bases — o modelo viu pouquíssimos exemplos.",
    ),
    "Lung Lesion": (
        "Lesão pulmonar",
        "Categoria guarda-chuva do CheXpert, que agrupa nódulo, massa e outras lesões focais.",
    ),
    "Fracture": (
        "Fratura",
        "Fratura óssea visível (arcos costais, clavícula, úmero).",
    ),
    "Lung Opacity": (
        "Opacidade pulmonar",
        "Termo guarda-chuva: qualquer aumento de densidade do parênquima, sem definir a causa.",
    ),
    "Enlarged Cardiomediastinum": (
        "Alargamento do mediastino",
        "Silhueta cardiomediastinal aumentada. Muito sensível à técnica (AP vs PA, distância, inspiração).",
    ),
}


def nome_pt(achado_en: str) -> str:
    """Traduz o rótulo em inglês usado pelo modelo para o termo clínico em português."""
    return ACHADOS.get(achado_en, (achado_en, ""))[0]


def explicacao(achado_en: str) -> str:
    """Devolve a linha de significado clínico do achado."""
    return ACHADOS.get(achado_en, ("", ""))[1]


def num(valor, casas: int = 2, sinal: bool = False) -> str:
    """
    Formata um número no padrão brasileiro: vírgula decimal, ponto de milhar.

    Python escreve 1234.5; em português escreve-se 1.234,5. Como o texto desta
    aula fala o tempo todo em "0,50", deixar as tabelas mostrando "0.50" criaria
    uma inconsistência boba justamente no ponto que a aula quer fixar.

    Passe sinal=True quando o sinal de "mais" importar (variações, diferenças).
    """
    formato = "{:" + ("+" if sinal else "") + ",." + str(casas) + "f}"
    texto = formato.format(valor)
    # A troca precisa de um passo intermediário: se trocássemos "," por "." e
    # depois "." por ",", desfaríamos a primeira troca.
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")



# ------------------------------------------------------------------------------
# Os modelos disponíveis.
#
# Todos têm a MESMA arquitetura (DenseNet-121) e a MESMA resolução de entrada
# (224x224). A única coisa que muda entre eles é EM QUE POPULAÇÃO foram
# treinados. Guarde isso: quando dois deles discordarem sobre a mesma
# radiografia, a diferença não é de "inteligência", é de origem dos dados.
# ------------------------------------------------------------------------------
MODELOS = {
    "densenet121-res224-all": {
        "rotulo": "Todas as bases combinadas (padrão)",
        "base": "NIH + PadChest + CheXpert + MIMIC-CXR + RSNA",
        "origem": "EUA e Espanha",
        "observacao": (
            "Treinado com tudo junto. É o mais generalista e o único que prevê "
            "os 18 achados. Use este como referência."
        ),
    },
    "densenet121-res224-nih": {
        "rotulo": "NIH ChestX-ray14",
        "base": "ChestX-ray14 — cerca de 112.000 imagens de 30.000 pacientes",
        "origem": "EUA (National Institutes of Health)",
        "observacao": (
            "Rótulos extraídos automaticamente de laudos por processamento de "
            "linguagem natural, com acurácia estimada em torno de 90%. Ou seja: "
            "aproximadamente 1 em cada 10 rótulos de treino está errado. Prevê 14 achados."
        ),
    },
    "densenet121-res224-chex": {
        "rotulo": "CheXpert (Stanford)",
        "base": "CheXpert — cerca de 224.000 imagens",
        "origem": "EUA (Stanford Hospital)",
        "observacao": (
            "Rótulos por NLP, com uma categoria explícita de incerteza que precisou "
            "ser tratada de alguma forma durante o treino. Inclui muitos exames em "
            "AP/decúbito, que distorcem a silhueta cardíaca."
        ),
    },
    "densenet121-res224-mimic_ch": {
        "rotulo": "MIMIC-CXR (rotulador CheXpert)",
        "base": "MIMIC-CXR — cerca de 370.000 imagens",
        "origem": "EUA (Beth Israel Deaconess, Boston)",
        "observacao": (
            "Base de UTI e emergência: muitos exames feitos no leito, com tubos, "
            "drenos e eletrodos no campo. Rótulos gerados pelo rotulador CheXpert."
        ),
    },
    "densenet121-res224-mimic_nb": {
        "rotulo": "MIMIC-CXR (rotulador NegBio)",
        "base": "MIMIC-CXR — exatamente as MESMAS imagens do modelo acima",
        "origem": "EUA (Beth Israel Deaconess, Boston)",
        "observacao": (
            "EXPERIMENTO IMPORTANTE: mesmas imagens, mesmos pacientes, mesma "
            "arquitetura. A única mudança foi o programa que leu os laudos para "
            "gerar os rótulos (NegBio em vez de CheXpert). Compare este modelo com "
            "o mimic_ch: a diferença que sobrar é culpa exclusivamente da rotulação."
        ),
    },
    "densenet121-res224-pc": {
        "rotulo": "PadChest",
        "base": "PadChest — cerca de 160.000 imagens",
        "origem": "Espanha (Hospital San Juan, Alicante)",
        "observacao": (
            "Única base grande fora dos EUA nesta lista, e a única com uma fatia "
            "considerável rotulada à mão por radiologistas. População, equipamento "
            "e prevalências diferentes das bases americanas."
        ),
    },
    "densenet121-res224-rsna": {
        "rotulo": "RSNA Pneumonia Challenge",
        "base": "RSNA Pneumonia Detection Challenge",
        "origem": "EUA (subconjunto da base NIH, re-anotado)",
        "observacao": (
            "Especialista: só foi treinado para pneumonia e opacidade pulmonar. "
            "Todos os outros achados saem vazios. É um bom exemplo de modelo de "
            "tarefa única, que é o formato mais comum dos produtos comerciais."
        ),
    },
}

MODELO_PADRAO = "densenet121-res224-all"

# ------------------------------------------------------------------------------
# Radiografias de exemplo, para a aula funcionar mesmo que você não tenha nenhuma
# imagem à mão. Vêm do repositório de testes do próprio TorchXRayVision.
# ------------------------------------------------------------------------------
URL_EXEMPLOS = "https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/"

IMAGENS_EXEMPLO = {
    "Tórax 1 (base NIH)": "00000001_000.png",
    "Tórax 2 (base NIH)": "00027426_000.png",
    "Pneumonia por COVID-19": "covid-19-pneumonia-58-prior.jpg",
    "Exemplo em DICOM (.dcm)": "1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm",
}

# Onde guardar as imagens baixadas. Preferimos a pasta do próprio arquivo; se ela
# não for gravável (acontece em alguns ambientes), caímos para a pasta temporária.
try:
    PASTA_EXEMPLOS = Path(__file__).resolve().parent / "imagens_exemplo"
except NameError:  # __file__ não existe em alguns notebooks
    PASTA_EXEMPLOS = Path.cwd() / "imagens_exemplo"

# Aviso que aparece em TODAS as páginas. A repetição é proposital.
AVISO_EDUCACIONAL = (
    "**Material didático.** Não é dispositivo médico, não tem registro na ANVISA "
    "e não pode ser usado para decisão clínica sobre nenhuma pessoa real. "
    "Não envie imagens de pacientes identificáveis."
)


# ==============================================================================
# BLOCO 2 — LENDO E PREPARANDO A IMAGEM
# ==============================================================================
# Esta é a parte que quase todo mundo pula e que causa quase todos os erros de
# quem usa IA em imagem pela primeira vez.
#
# Uma radiografia digital não é "uma foto". Ela é uma matriz de números que
# representam atenuação de raios X. Um DICOM de tórax costuma ter 12 a 16 bits
# por pixel (até 65.536 níveis de cinza); o seu monitor mostra 8 bits (256
# níveis). Quando alguém exporta um DICOM para PNG ou JPG, essa conversão já
# jogou informação fora — e o modelo vai trabalhar com o que sobrou.
#
# O TorchXRayVision padroniza tudo para uma escala de [-1024, +1024]. Todos os
# modelos desta aula assumem essa escala. Se você mandar uma imagem em [0, 255]
# direto para a rede, ela não dá erro: ela devolve um resultado errado com toda
# a confiança do mundo. Esse é o tipo de falha silenciosa que você precisa
# aprender a reconhecer.
# ==============================================================================

# Resolução de entrada dos modelos DenseNet desta biblioteca.
RESOLUCAO = 224


@st.cache_data(show_spinner="Baixando as radiografias de exemplo…")
def baixar_imagens_exemplo() -> dict:
    """
    Baixa (uma única vez) as radiografias de exemplo para uma pasta local.

    O decorador @st.cache_data faz o Streamlit guardar o resultado: da segunda
    vez em diante a função nem chega a ser executada. Sem isso, o Streamlit
    reexecuta o arquivo inteiro a cada clique e você baixaria tudo de novo.
    """
    caminhos = {}
    try:
        PASTA_EXEMPLOS.mkdir(parents=True, exist_ok=True)
        pasta = PASTA_EXEMPLOS
    except OSError:
        # Pasta do script não é gravável (ex.: drive de rede em modo leitura).
        pasta = Path(tempfile.gettempdir()) / "imagens_exemplo_rx"
        pasta.mkdir(parents=True, exist_ok=True)

    for rotulo, arquivo in IMAGENS_EXEMPLO.items():
        destino = pasta / arquivo
        if not destino.exists():
            try:
                urllib.request.urlretrieve(URL_EXEMPLOS + arquivo, destino)
            except Exception:
                # Sem internet ou arquivo indisponível: simplesmente não
                # oferecemos essa imagem, em vez de derrubar o aplicativo.
                continue
        caminhos[rotulo] = str(destino)
    return caminhos


def salvar_upload(arquivo_enviado) -> str:
    """
    Grava em disco o arquivo que o aluno enviou pelo navegador.

    O Streamlit entrega o upload como bytes na memória, mas tanto o leitor de
    DICOM quanto o de PNG precisam de um caminho de arquivo. Então gravamos num
    arquivo temporário, preservando a extensão original (que é o que nos permite
    reconhecer um .dcm).
    """
    sufixo = Path(arquivo_enviado.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as temporario:
        temporario.write(arquivo_enviado.getbuffer())
        return temporario.name


def carregar_imagem(caminho):
    """
    Lê uma radiografia de disco e devolve (imagem, informações).

    Retorno:
      imagem  : np.ndarray de forma [1, altura, largura], valores em [-1024, 1024]
                (o "1" da frente é o canal de cor: radiografia é monocromática)
      info    : dicionário com o que descobrimos sobre o arquivo, para exibir

    Por que não usamos xrv.utils.load_image()?
      Porque ela assume 8 bits por pixel e levanta erro em PNG de 16 bits, que é
      um formato comum de exportação de radiografia real. A função abaixo faz a
      mesma coisa, mas detectando a profundidade de bits.
    """
    caminho = str(caminho)

    # --- É DICOM? ------------------------------------------------------------
    # O padrão DICOM manda 128 bytes de preâmbulo seguidos das 4 letras "DICM".
    # É assim que se identifica o arquivo sem depender da extensão (muitos DICOM
    # reais vêm sem extensão nenhuma).
    with open(caminho, "rb") as arquivo:
        cabecalho = arquivo.read(132)
    tem_marca_dicm = len(cabecalho) >= 132 and cabecalho[128:132] == b"DICM"

    if tem_marca_dicm or caminho.lower().endswith(".dcm"):
        try:
            # voi_lut=True aplica a janela (window/level) que está gravada no
            # cabeçalho do DICOM — é a mesma janela que o radiologista vê no PACS.
            bruta = xrv.utils.read_xray_dcm(caminho, voi_lut=True, fix_monochrome=True)
            janela = "aplicada a partir do cabeçalho (VOI LUT)"
        except Exception:
            # Nem todo DICOM traz VOI LUT. Sem ela, usamos os valores crus.
            bruta = xrv.utils.read_xray_dcm(caminho, voi_lut=False, fix_monochrome=True)
            janela = "não disponível no arquivo; usados os valores crus"

        imagem = np.asarray(bruta, dtype=np.float32)[None, ...]
        info = {
            "formato": "DICOM",
            "profundidade": "12–16 bits (típico de DICOM)",
            "janela": janela,
            "dimensoes": f"{imagem.shape[2]} x {imagem.shape[1]} pixels",
            "canais": 1,
        }
        return imagem, info

    # --- PNG / JPG / TIFF ----------------------------------------------------
    bruta = skimage.io.imread(caminho)

    canais_originais = 1
    if bruta.ndim == 3:
        # Imagem salva em RGB (ou RGBA). Radiografia é cinza, mas ao exportar
        # para PNG/JPG ela costuma ser triplicada nos três canais. Tiramos a
        # média dos canais de cor para voltar a um canal só.
        canais_originais = bruta.shape[2]
        bruta = bruta[..., :3].mean(axis=2)

    # Descobrir a profundidade de bits para normalizar corretamente.
    if bruta.dtype == np.uint8:
        maximo = 255.0
        profundidade = "8 bits (0–255)"
    elif np.issubdtype(bruta.dtype, np.integer):
        # 16 bits: usamos o máximo REAL da imagem, e não 65535. Usar 65535 numa
        # radiografia cujo pixel mais claro vale 4000 deixaria tudo quase preto.
        maximo = float(bruta.max())
        profundidade = f"{bruta.dtype} — máximo observado: {int(maximo)}"
    else:
        maximo = float(bruta.max())
        profundidade = f"ponto flutuante ({bruta.dtype})"

    if maximo <= 0:
        raise ValueError(
            "A imagem parece estar completamente preta (valor máximo igual a zero)."
        )

    # normalize() leva a imagem para a escala [-1024, 1024] que os modelos esperam.
    imagem = xrv.datasets.normalize(bruta, maximo)
    imagem = np.asarray(imagem, dtype=np.float32)[None, ...]

    info = {
        "formato": Path(caminho).suffix.upper().lstrip(".") or "imagem",
        "profundidade": profundidade,
        "janela": "não se aplica (arquivo já vem convertido para exibição)",
        "dimensoes": f"{imagem.shape[2]} x {imagem.shape[1]} pixels",
        "canais": canais_originais,
    }
    return imagem, info


def para_visualizacao(imagem) -> np.ndarray:
    """
    Converte a matriz do modelo, em [-1024, 1024], para algo que a tela exibe:
    uma matriz 2D com valores entre 0 (preto) e 1 (branco).
    """
    plano = imagem[0] if imagem.ndim == 3 else imagem
    return np.clip((plano + 1024.0) / 2048.0, 0.0, 1.0)


def recortar_centro(imagem) -> np.ndarray:
    """
    Recorte quadrado central.

    A rede só aceita imagens quadradas. Como radiografias de tórax costumam ser
    mais altas que largas, o recorte tira faixas de cima e de baixo.

    CONSEQUÊNCIA CLÍNICA: os ápices pulmonares e os seios costofrênicos são,
    justamente, as bordas. Se a imagem estiver mal enquadrada, um pneumotórax
    apical ou um derrame pequeno podem ser literalmente cortados fora antes de o
    modelo ver qualquer coisa. Preste atenção nisso no módulo 1.
    """
    return xrv.datasets.XRayCenterCrop()(imagem)


def redimensionar(imagem, tamanho: int = RESOLUCAO) -> np.ndarray:
    """
    Reduz a imagem para tamanho x tamanho (224x224 por padrão).

    Pare um segundo para pensar no tamanho disso: uma radiografia de tórax
    original tem tipicamente 2000x2500 pixels. Estamos jogando fora mais de 98%
    dos pixels. Um nódulo de 4 mm ocupa menos de um pixel na imagem reduzida.

    É por isso que estes modelos são razoáveis para achados grandes (derrame,
    cardiomegalia, consolidação extensa) e ruins para achados finos.
    """
    return xrv.datasets.XRayResizer(tamanho)(imagem)


def preprocessar(imagem):
    """Aplica o recorte e a redução, devolvendo as duas etapas para exibição."""
    recortada = recortar_centro(imagem)
    reduzida = redimensionar(recortada)
    return recortada, reduzida


def para_tensor(imagem) -> torch.Tensor:
    """
    Converte a matriz numpy [1, A, L] no tensor [1, 1, A, L] que o PyTorch espera.

    A dimensão extra na frente é o "lote" (batch): redes neurais processam vários
    exames de uma vez. Aqui o lote tem tamanho 1, porque estamos vendo um exame
    só — mas a dimensão precisa existir mesmo assim.
    """
    matriz = np.ascontiguousarray(imagem, dtype=np.float32)
    return torch.from_numpy(matriz)[None, ...]


# ==============================================================================
# BLOCO 3 — CARREGAR O MODELO E RODAR A PREVISÃO
# ==============================================================================
# Aqui está o coração da aula, e também a armadilha mais importante dela.
#
# QUANDO O MODELO DEVOLVE 0,73, ISSO NÃO QUER DIZER "73% DE CHANCE DE TER A
# DOENÇA". Vale a pena entender exatamente o que aquele número é.
#
# A rede termina calculando, para cada achado, um número sem limite superior nem
# inferior, chamado "logito". Para espremer esse número entre 0 e 1, aplica-se a
# função sigmoide. O resultado já parece uma probabilidade — mas é uma
# probabilidade MAL CALIBRADA: como as doenças são raras nas bases de treino, os
# valores ficam quase todos amontoados perto de zero. Um derrame pleural
# evidente pode sair com sigmoide de 0,08.
#
# Para contornar isso, o TorchXRayVision guarda, junto com cada modelo, um
# "ponto de operação" por achado (op_threshs): o valor de corte escolhido na
# validação. Aí ele reescala a saída de modo que o ponto de operação vire
# exatamente 0,50. Essa reescala é a função op_norm().
#
# Ou seja, a saída que você vê significa:
#     > 0,50  →  acima do ponto de corte do modelo para esse achado
#     < 0,50  →  abaixo do ponto de corte
#     = 0,73  →  "bem acima do corte", e NÃO "73% de probabilidade"
#
# Neste bloco mostramos os DOIS números lado a lado, justamente para você ver a
# diferença com os próprios olhos.
# ==============================================================================


def _construir_modelo(construtor, descricao: str):
    """
    Constrói um modelo, transformando a falha mais comum numa mensagem útil.

    Por que isto existe: se o download dos pesos for interrompido no meio — queda
    de internet, Ctrl+C, aba fechada, sala de aula inteira baixando ao mesmo
    tempo — o arquivo fica pela metade dentro da pasta de cache. E aí ele NÃO se
    conserta sozinho: todas as tentativas seguintes falham, sempre, com um erro
    de desserialização que não diz nada sobre a causa real.

    Como isso é praticamente garantido de acontecer com alguém numa turma, vale
    a pena capturar o erro e dizer qual pasta apagar.
    """
    try:
        modelo = construtor()
    except Exception as erro:
        try:
            pasta = xrv.utils.get_cache_dir()
        except Exception:
            pasta = "~/.torchxrayvision/models_data"
        raise RuntimeError(
            f"Não consegui carregar {descricao}.\n\n"
            "A causa mais provável é um download interrompido, que deixou o "
            "arquivo de pesos pela metade. Apague o conteúdo da pasta de cache "
            "abaixo e abra este módulo de novo — o download recomeça do zero:\n\n"
            f"`{pasta}`\n\n"
            f"Detalhe técnico: `{erro}`"
        ) from erro

    # eval() coloca a rede em modo de avaliação. Isso importa de verdade: desliga
    # camadas como dropout e congela a normalização em lote, que se comportam de
    # um jeito durante o treino e de outro durante o uso. Esquecer o .eval() é um
    # erro clássico e faz a mesma imagem dar resultados diferentes a cada execução.
    modelo.eval()
    return modelo


@st.cache_resource(show_spinner=False)
def carregar_modelo(pesos: str):
    """
    Baixa (na primeira vez) e carrega um modelo DenseNet-121 pré-treinado.

    @st.cache_resource guarda o modelo já carregado na memória do servidor. Sem
    ele, cada clique na interface recarregaria 30 MB de pesos do disco.
    """
    return _construir_modelo(
        lambda: xrv.models.DenseNet(weights=pesos),
        f"o modelo `{pesos}`",
    )


def achados_do_modelo(modelo):
    """
    Devolve [(índice, nome em inglês), ...] apenas dos achados que ESTE modelo
    realmente prevê.

    Detalhe técnico que vale conhecer: todos os modelos têm 18 saídas, mas os
    treinados numa base específica deixam vazio (string "") o nome dos achados
    que aquela base não anotava. O modelo RSNA, por exemplo, só tem rótulo em
    dois dos 18. Se você não filtrar isso, acaba exibindo números que não
    significam absolutamente nada — eles são apenas o valor neutro 0,50.
    """
    return [(i, nome) for i, nome in enumerate(modelo.pathologies) if nome]


def inferir(modelo, imagem224) -> pd.DataFrame:
    """
    Roda a rede sobre a imagem já pré-processada e devolve uma tabela.

    Colunas devolvidas:
      achado        : nome clínico em português
      achado_en     : rótulo original, como aparece na literatura
      saida         : saída do modelo, onde 0,50 é o ponto de operação
      sigmoide      : a probabilidade crua, antes da reescala
      limiar        : o ponto de operação bruto daquele achado
    """
    entrada = para_tensor(imagem224)

    # torch.no_grad() diz ao PyTorch para não guardar o histórico de operações.
    # Só precisamos disso para treinar; na hora de usar, economiza memória.
    with torch.no_grad():
        # (a) A saída "oficial": forward() já aplica sigmoide e depois op_norm.
        saida = modelo(entrada)[0]

        # (b) A saída crua: reproduzimos o final da rede na mão para conseguir o
        #     logito antes da reescala. features2() faz a extração de
        #     características e a média espacial; classifier é a camada linear final.
        vetor = modelo.features2(entrada)
        logitos = modelo.classifier(vetor)[0]
        sigmoides = torch.sigmoid(logitos)

    limiares = getattr(modelo, "op_threshs", None)

    linhas = []
    for indice, nome in achados_do_modelo(modelo):
        limiar = float("nan")
        if limiares is not None:
            limiar = float(limiares[indice])
            # NaN no limiar = achado que este modelo não sabe prever. Nesse caso
            # op_norm devolve exatamente 0,50, um valor sem significado clínico.
            if math.isnan(limiar):
                continue

        linhas.append(
            {
                "achado": nome_pt(nome),
                "achado_en": nome,
                "saida": float(saida[indice]),
                "sigmoide": float(sigmoides[indice]),
                "limiar": limiar,
            }
        )

    tabela = pd.DataFrame(linhas)
    if not tabela.empty:
        tabela = tabela.sort_values("saida", ascending=False).reset_index(drop=True)
    return tabela


# ==============================================================================
# BLOCO 4 — ONDE O MODELO OLHOU (mapa de ativação)
# ==============================================================================
# "Explicabilidade" virou palavra de ordem em IA médica. A técnica mais comum em
# imagem é o mapa de ativação de classe (CAM), e a boa notícia é que, para esta
# arquitetura específica, ele é EXATO — não é uma aproximação.
#
# Por quê? Porque o DenseNet termina assim:
#
#     imagem → [muitas camadas] → 1024 mapas de 7x7 → média de cada mapa
#            → 1024 números → camada linear (pesos w) → 18 logitos
#
# Como o último passo antes da camada linear é uma média espacial simples, dá
# para trocar a ordem das operações. O logito do achado c é:
#
#     logito_c = média_espacial( soma_k w_ck * mapa_k ) + viés_c
#
# O termo dentro da média — soma_k w_ck * mapa_k — é uma imagem de 7x7 que diz,
# posição por posição, o quanto aquela região empurrou o logito para cima ou
# para baixo. É isso que ampliamos e sobrepomos à radiografia.
#
# LEIA COM CUIDADO ESTE AVISO: um mapa de calor que cai em cima do pulmão certo
# NÃO prova que o modelo raciocinou corretamente. Ele só mostra QUAIS PIXELS
# influenciaram a conta. Modelos que usaram atalhos (um dreno de tórax, uma marca
# de posicionamento, o texto "PORTABLE" gravado no canto) frequentemente geram
# mapas que parecem plausíveis. Explicabilidade é ferramenta de auditoria, não é
# certificado de qualidade.
# ==============================================================================


def mapa_de_ativacao(modelo, imagem224, indice_achado: int):
    """
    Calcula o mapa de ativação de classe para um achado.

    Devolve (mapa_ampliado, logito, conferencia), onde:
      mapa_ampliado : matriz 224x224 com a contribuição de cada região
      logito        : o logito daquele achado
      conferencia   : média do mapa + viés, que deve bater com o logito
                      (usamos isso no aplicativo para provar que a conta fecha)
    """
    entrada = para_tensor(imagem224)

    with torch.no_grad():
        # modelo.features é o extrator convolucional. A saída tem forma
        # [1, 1024, 7, 7]: 1024 detectores de padrão, cada um com um mapa 7x7.
        mapas = modelo.features(entrada)

        # A mesma ReLU que features2() aplica antes da média espacial. Precisa
        # estar aqui, senão a conta não fecha.
        mapas = torch.relu(mapas)[0]  # [1024, 7, 7]

        # Os pesos que a camada final usa para este achado específico.
        pesos = modelo.classifier.weight[indice_achado]  # [1024]
        vies = float(modelo.classifier.bias[indice_achado])

        # Combinação linear ponderada dos 1024 mapas -> um único mapa 7x7.
        mapa = (mapas * pesos[:, None, None]).sum(dim=0)  # [7, 7]

        conferencia = float(mapa.mean()) + vies

        # E o logito calculado pelo caminho normal, para comparar.
        vetor = modelo.features2(entrada)
        logito = float(modelo.classifier(vetor)[0][indice_achado])

    mapa = mapa.numpy()

    # Ampliar de 7x7 para 224x224. A interpolação suaviza e dá aquele aspecto de
    # "mancha" — o que também é um lembrete de que a resolução real da explicação
    # é grosseira: cada quadradinho do mapa cobre 32x32 pixels da imagem, algo em
    # torno de 5 cm de tórax.
    ampliado = skimage.transform.resize(
        mapa, (RESOLUCAO, RESOLUCAO), order=3, mode="constant", preserve_range=True
    )
    return ampliado, logito, conferencia


# ==============================================================================
# BLOCO 5 — LIMIAR, SENSIBILIDADE E O TEOREMA DE BAYES
# ==============================================================================
# Se você só levar uma coisa desta aula, que seja esta.
#
# Um modelo de IA é um teste diagnóstico. Como qualquer teste diagnóstico, ele
# tem sensibilidade e especificidade — e essas duas propriedades NÃO dependem da
# prevalência. Mas o que interessa ao médico à beira do leito não é nenhuma das
# duas: é o valor preditivo positivo (VPP), a probabilidade de a pessoa
# realmente ter a doença dado que o teste deu positivo. E o VPP depende, e muito,
# da prevalência.
#
# Consequência prática, que aparece o tempo todo em IA médica: um modelo
# publicado com AUC de 0,95 num hospital terciário, onde 20% dos exames têm o
# achado, pode virar uma máquina de falsos positivos quando instalado numa UBS
# onde a prevalência é 1%. O modelo não piorou. A população mudou.
#
# É por isso que "acurácia de 95%" é uma informação quase inútil sozinha, e por
# isso que você deve sempre perguntar: validado em quem, com que prevalência?
# ==============================================================================


def razoes_de_verossimilhanca(sensibilidade: float, especificidade: float):
    """
    Calcula as razões de verossimilhança positiva e negativa.

        RV+ = sensibilidade / (1 - especificidade)
        RV- = (1 - sensibilidade) / especificidade

    Vale a pena decorar a leitura clínica delas:
        RV+ > 10  → altera bastante a conduta a favor do diagnóstico
        RV+ 5–10  → altera moderadamente
        RV+ 2–5   → altera pouco
        RV+ ~1    → o teste não serviu para nada

    A grande vantagem da razão de verossimilhança sobre a sensibilidade e a
    especificidade isoladas é que ela combina direto com a probabilidade
    pré-teste, sem precisar saber a prevalência da população inteira.
    """
    # Prendemos os valores longe de 0 e 1 para não dividir por zero quando
    # alguém arrasta o controle deslizante até o extremo.
    sensibilidade = min(max(sensibilidade, 1e-4), 1 - 1e-4)
    especificidade = min(max(especificidade, 1e-4), 1 - 1e-4)

    rv_positiva = sensibilidade / (1.0 - especificidade)
    rv_negativa = (1.0 - sensibilidade) / especificidade
    return rv_positiva, rv_negativa


def probabilidade_pos_teste(pre_teste: float, sensibilidade: float, especificidade: float):
    """
    Aplica o teorema de Bayes na forma de chances (odds), que é a mais simples:

        chance pós-teste = chance pré-teste x razão de verossimilhança

    Devolve (VPP, 1 - VPN), ou seja:
      - a probabilidade de doença SE o modelo apontou positivo
      - a probabilidade de doença SE o modelo apontou negativo
    """
    pre_teste = min(max(pre_teste, 1e-6), 1 - 1e-6)
    rv_positiva, rv_negativa = razoes_de_verossimilhanca(sensibilidade, especificidade)

    chance_pre = pre_teste / (1.0 - pre_teste)

    chance_pos_positivo = chance_pre * rv_positiva
    chance_pos_negativo = chance_pre * rv_negativa

    vpp = chance_pos_positivo / (1.0 + chance_pos_positivo)
    doenca_apesar_de_negativo = chance_pos_negativo / (1.0 + chance_pos_negativo)
    return vpp, doenca_apesar_de_negativo


# ==============================================================================
# BLOCO 6 — TESTE DE ROBUSTEZ
# ==============================================================================
# Um radiologista humano que recebe uma radiografia girada 10 graus dá o mesmo
# laudo. Um humano que recebe a mesma radiografia um pouco mais escura dá o mesmo
# laudo. Redes neurais não têm essa garantia.
#
# As funções abaixo aplicam perturbações que, do ponto de vista clínico, NÃO
# deveriam mudar nada. Se a saída do modelo mudar muito, você acabou de descobrir
# uma fragilidade — e um motivo concreto para exigir teste local antes de
# implantar qualquer sistema desses no seu serviço.
#
# Preste atenção especial ao espelhamento horizontal. Ele troca direita por
# esquerda. Se o modelo mantiver a mesma pontuação de cardiomegalia numa imagem
# espelhada (onde o coração agora aparece à direita), ele está ignorando
# lateralidade — o que seria um erro grosseiro para um humano.
# ==============================================================================

PERTURBACOES = {
    "Nenhuma (imagem original)": ("nenhuma", 0.0),
    "Rotação de 5 graus": ("rotacao", 5.0),
    "Rotação de 15 graus": ("rotacao", 15.0),
    "Espelhamento horizontal (troca direita e esquerda)": ("espelho", 0.0),
    "Mais clara (gama 0,6)": ("gama", 0.6),
    "Mais escura (gama 1,7)": ("gama", 1.7),
    "Oclusão do quadrante inferior direito": ("oclusao", 0.0),
    "Ruído leve": ("ruido", 60.0),
}


def perturbar(imagem, tipo: str, intensidade: float) -> np.ndarray:
    """
    Aplica uma perturbação à imagem em [-1024, 1024] e devolve a versão alterada.

    A imagem de entrada e a de saída têm a mesma forma [1, altura, largura].
    """
    plano = imagem[0]

    if tipo == "nenhuma":
        alterado = plano.copy()

    elif tipo == "rotacao":
        # cval=-1024 preenche os cantos vazios com preto, na nossa escala.
        alterado = skimage.transform.rotate(
            plano, intensidade, mode="constant", cval=-1024.0, preserve_range=True
        )

    elif tipo == "espelho":
        # O ::-1 inverte a ordem das colunas. O .copy() é necessário porque o
        # PyTorch não aceita matrizes com passo negativo.
        alterado = plano[:, ::-1].copy()

    elif tipo == "gama":
        # Correção de gama: leva para [0,1], eleva à potência, volta para a escala.
        # Gama < 1 clareia, gama > 1 escurece. É basicamente mexer no brilho do
        # negatoscópio, algo que acontece o tempo todo na vida real.
        normalizado = np.clip((plano + 1024.0) / 2048.0, 0.0, 1.0)
        alterado = (normalizado ** intensidade) * 2048.0 - 1024.0

    elif tipo == "oclusao":
        # Apaga um quarto da imagem. Simula, de forma grosseira, uma etiqueta de
        # chumbo, um artefato de processamento ou um campo cortado.
        alterado = plano.copy()
        altura, largura = alterado.shape
        alterado[altura // 2 :, largura // 2 :] = -1024.0

    elif tipo == "ruido":
        # Ruído gaussiano, como o de uma radiografia feita com baixa dose.
        gerador = np.random.default_rng(seed=0)  # semente fixa: resultado reproduzível
        alterado = plano + gerador.normal(0.0, intensidade, size=plano.shape)
        alterado = np.clip(alterado, -1024.0, 1024.0)

    else:
        raise ValueError(f"Perturbação desconhecida: {tipo}")

    return np.asarray(alterado, dtype=np.float32)[None, ...]


# ==============================================================================
# BLOCO 7 — GRÁFICOS
# ==============================================================================
# Nada de conceito novo aqui: são só funções de desenho, separadas do resto para
# o código dos módulos ficar legível.
# ==============================================================================

# Paleta em tons discretos, para o gráfico não competir com a radiografia.
COR_ACIMA = "#c0392b"     # vermelho: acima do ponto de corte
COR_ABAIXO = "#7f8c8d"    # cinza: abaixo do ponto de corte
COR_DESTAQUE = "#2c6fbb"  # azul


def grafico_barras(tabela: pd.DataFrame, limiar: float):
    """Gráfico de barras horizontais com a saída do modelo para cada achado."""
    figura, eixo = plt.subplots(figsize=(7, 0.34 * len(tabela) + 1.2))

    ordenada = tabela.sort_values("saida")
    cores = [COR_ACIMA if v >= limiar else COR_ABAIXO for v in ordenada["saida"]]

    eixo.barh(ordenada["achado"], ordenada["saida"], color=cores)
    eixo.axvline(limiar, color="black", linestyle="--", linewidth=1.2)
    eixo.text(
        limiar,
        len(ordenada) - 0.3,
        f"  limiar = {num(limiar)}",
        va="center",
        fontsize=9,
    )

    eixo.set_xlim(0, 1)
    eixo.set_xlabel("Saída do modelo (0,50 = ponto de operação, NÃO é probabilidade)")
    eixo.spines[["top", "right"]].set_visible(False)
    eixo.tick_params(labelsize=9)
    figura.tight_layout()
    return figura


def figura_sobreposicao(imagem224, mapa, titulo: str):
    """
    Desenha a radiografia com o mapa de ativação por cima.

    Usamos um mapa de cores divergente centrado em zero, e não o clássico
    "quanto mais vermelho mais importante". A razão é honestidade: o mapa tem
    valores positivos E negativos, e essa informação é útil.
        vermelho → região que empurrou a favor do achado
        azul     → região que empurrou contra o achado
        neutro   → região que não pesou
    """
    figura, eixos = plt.subplots(1, 2, figsize=(9, 4.6))

    fundo = para_visualizacao(imagem224)

    eixos[0].imshow(fundo, cmap="gray")
    eixos[0].set_title("Imagem que o modelo recebeu", fontsize=10)
    eixos[0].axis("off")

    limite = float(np.abs(mapa).max()) or 1.0
    normalizacao = TwoSlopeNorm(vmin=-limite, vcenter=0.0, vmax=limite)

    eixos[1].imshow(fundo, cmap="gray")
    imagem_calor = eixos[1].imshow(mapa, cmap="coolwarm", norm=normalizacao, alpha=0.45)
    eixos[1].set_title(titulo, fontsize=10)
    eixos[1].axis("off")

    barra = figura.colorbar(imagem_calor, ax=eixos[1], fraction=0.046, pad=0.04)
    barra.set_label("contra  ←  contribuição  →  a favor", fontsize=8)
    barra.ax.tick_params(labelsize=7)

    figura.tight_layout()
    return figura


def grafico_bayes(sensibilidade: float, especificidade: float, pre_teste_marcado: float):
    """
    Desenha o VPP em função da prevalência, marcando o cenário escolhido.

    Este é, provavelmente, o gráfico mais importante do arquivo inteiro.
    """
    prevalencias = np.logspace(-3, np.log10(0.6), 300)  # de 0,1% a 60%
    vpps = np.array(
        [probabilidade_pos_teste(p, sensibilidade, especificidade)[0] for p in prevalencias]
    )

    figura, eixo = plt.subplots(figsize=(7, 4.2))
    eixo.plot(prevalencias * 100, vpps * 100, color=COR_DESTAQUE, linewidth=2.2)

    vpp_marcado = probabilidade_pos_teste(pre_teste_marcado, sensibilidade, especificidade)[0]
    eixo.plot([pre_teste_marcado * 100], [vpp_marcado * 100], "o", color=COR_ACIMA, markersize=9)
    eixo.annotate(
        f"seu cenário\nVPP = {num(vpp_marcado * 100, 1)}%",
        xy=(pre_teste_marcado * 100, vpp_marcado * 100),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=9,
        color=COR_ACIMA,
    )

    # Cenários de referência, para dar noção de escala ao aluno.
    for prevalencia, rotulo in [
        (0.005, "rastreamento\npopulacional"),
        (0.05, "ambulatório"),
        (0.25, "pronto-socorro /\nUTI"),
    ]:
        eixo.axvline(prevalencia * 100, color="#bbbbbb", linestyle=":", linewidth=1)
        eixo.text(
            prevalencia * 100, 3, rotulo, fontsize=7.5, color="#666666", ha="center"
        )

    eixo.set_xscale("log")
    eixo.set_xlabel("Prevalência do achado na população testada (%, escala logarítmica)")
    eixo.set_ylabel("VPP: chance de realmente ter a doença\ndado que o modelo apontou positivo (%)")
    eixo.set_ylim(0, 100)
    eixo.grid(alpha=0.25)
    eixo.spines[["top", "right"]].set_visible(False)
    figura.tight_layout()
    return figura


# ==============================================================================
# BLOCO 8 — A INTERFACE (Streamlit)
# ==============================================================================
# Daqui para baixo é só interface. O Streamlit funciona de um jeito que confunde
# quem vem de outras linguagens: a CADA clique, ele reexecuta este arquivo
# inteiro, de cima a baixo. Não existe "evento de botão". É isso que torna os
# decoradores de cache (@st.cache_data e @st.cache_resource) obrigatórios, e não
# um luxo de otimização.
# ==============================================================================

st.set_page_config(
    page_title="IA em Radiologia — aula prática",
    layout="wide",
)

# ------------------------------------------------------------------------------
# A aula tem oito módulos, e cada um é precedido por uma página de PREPARAÇÃO
# que ensina o vocabulário necessário para acompanhá-lo.
#
# A razão de existirem: esta aula é para estudantes de medicina, que não têm por
# que já saber o que é um logito, um rótulo ou uma arquitetura. Sem as páginas de
# preparação, o aluno atravessa a aula decorando números em vez de entendê-los.
#
# Os títulos ficam todos nesta lista única, e a barra lateral e o roteador são
# construídos a partir dela — assim não há como um ficar fora de sincronia com o
# outro quando você editar, remover ou reordenar alguma coisa.
# ------------------------------------------------------------------------------
AULA = [
    ("1", "O que o modelo realmente vê"),
    ("2", "Lendo a saída do modelo"),
    ("3", "Limiar, Bayes e valor preditivo"),
    ("4", "Onde o modelo olhou"),
    ("5", "Modelos discordam entre si"),
    ("6", "Teste de robustez"),
    ("7", "Viés: o que a rede aprendeu sem querer"),
    ("8", "Segmentação anatômica"),
]

PAGINA_INICIO = "Início"


def rotulo_preparacao(numero: str) -> str:
    """Nome da página de preparação na barra lateral."""
    return f"{numero} · Preparação"


def rotulo_modulo(numero: str, titulo: str) -> str:
    """Nome do módulo na barra lateral."""
    return f"{numero} · {titulo}"


# A lista que a barra lateral mostra: preparação e módulo, alternando.
MODULOS = [PAGINA_INICIO]
for _numero, _titulo in AULA:
    MODULOS.append(rotulo_preparacao(_numero))
    MODULOS.append(rotulo_modulo(_numero, _titulo))

# Atalhos para o roteador descobrir, a partir do rótulo clicado, de que página
# se trata — sem precisar comparar textos soltos espalhados pelo código.
NUMERO_DA_PREPARACAO = {rotulo_preparacao(n): n for n, _ in AULA}
NUMERO_DO_MODULO = {rotulo_modulo(n, t): n for n, t in AULA}
TITULO_DO_MODULO = dict(AULA)


# ------------------------------------------------------------------------------
# Funções com cache. Repare que elas apenas embrulham as funções "de verdade"
# definidas lá em cima. Manter a lógica separada da interface é o que permite
# você reaproveitar este código num script seu, sem o Streamlit junto.
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Lendo e preparando a radiografia…")
def preparar(caminho: str):
    """Lê a imagem do disco e devolve todas as etapas do pré-processamento."""
    imagem, info = carregar_imagem(caminho)
    recortada, reduzida = preprocessar(imagem)
    return imagem, info, recortada, reduzida


@st.cache_data(show_spinner=False)
def inferir_cache(pesos: str, imagem224: np.ndarray) -> pd.DataFrame:
    """Roda a previsão, guardando o resultado para não recalcular a cada clique."""
    return inferir(carregar_modelo(pesos), imagem224)


def para_exibir(imagem) -> np.ndarray:
    """Converte para uint8 (0–255), que é o formato que st.image() espera."""
    return (para_visualizacao(imagem) * 255).astype(np.uint8)


def cabecalho(titulo: str, subtitulo: str = ""):
    """Cabeçalho padronizado de cada módulo, com o aviso educacional."""
    st.title(titulo)
    if subtitulo:
        st.caption(subtitulo)
    st.info(AVISO_EDUCACIONAL, icon="⚠️")


# ------------------------------------------------------------------------------
# Barra lateral: escolha do módulo e da radiografia.
# ------------------------------------------------------------------------------
def barra_lateral():
    """Desenha a barra lateral e devolve (módulo escolhido, caminho da imagem)."""
    st.sidebar.title("Aula: IA em Radiologia")
    st.sidebar.caption("Siga os módulos em ordem na primeira vez.")

    modulo = st.sidebar.radio("Módulo", MODULOS, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.subheader("Radiografia")

    origem = st.sidebar.radio(
        "De onde vem a imagem?",
        ["Exemplos da aula", "Enviar uma imagem minha"],
    )

    caminho = None

    if origem == "Exemplos da aula":
        exemplos = baixar_imagens_exemplo()
        if not exemplos:
            st.sidebar.error(
                "Não consegui baixar as imagens de exemplo. Verifique sua conexão "
                "ou envie uma imagem sua na opção acima."
            )
        else:
            escolha = st.sidebar.selectbox("Escolha o exame", list(exemplos.keys()))
            caminho = exemplos[escolha]
    else:
        st.sidebar.warning(
            "Use apenas imagens públicas ou anonimizadas. Nunca envie exames de "
            "pacientes identificáveis para um programa de demonstração.",
            icon="🔒",
        )
        enviado = st.sidebar.file_uploader(
            "Radiografia de tórax",
            type=["png", "jpg", "jpeg", "dcm", "tif", "tiff"],
        )
        if enviado is not None:
            caminho = salvar_upload(enviado)

    st.sidebar.divider()
    st.sidebar.caption(
        "TorchXRayVision · DenseNet-121\n\n"
        "Material didático. Sem valor diagnóstico."
    )
    return modulo, caminho


# ------------------------------------------------------------------------------
# MÓDULO: Início
# ------------------------------------------------------------------------------
def pagina_inicio():
    cabecalho(
        "Inteligência artificial em radiologia, na prática",
        "Um roteiro em 8 módulos para quem vai ser médico, não engenheiro.",
    )

    st.markdown(
        """
### O que você vai fazer aqui

Você vai operar modelos de verdade — os mesmos pesos publicados por grupos de
Stanford, do NIH e do MIT — sobre radiografias de tórax, e vai testar cada
promessa que se costuma fazer sobre eles.

A ordem dos módulos não é aleatória. Ela reproduz o caminho que um exame faz
dentro de um sistema de IA, e cada etapa esconde um modo de falhar:

| Módulo | Pergunta que ele responde |
|---|---|
| **1 · O que o modelo vê** | A imagem que a rede recebe é a mesma que você olha no negatoscópio? |
| **2 · Lendo a saída** | O que significa, exatamente, o número que aparece na tela? |
| **3 · Limiar e Bayes** | Por que um modelo excelente pode ser inútil no seu serviço? |
| **4 · Onde olhou** | Dá para auditar a decisão? Até onde essa auditoria vale? |
| **5 · Discordância** | Dois modelos, mesma imagem, respostas diferentes. E agora? |
| **6 · Robustez** | O laudo muda se eu girar a imagem 15 graus? |
| **7 · Viés** | O que a rede aprendeu que ninguém pediu para ela aprender? |
| **8 · Segmentação** | Classificar não é a única coisa que IA faz com imagem. |

### Como usar

1. Escolha uma radiografia na barra lateral (as de exemplo baixam sozinhas).
2. Percorra os módulos em ordem.
3. Em cada módulo, leia o texto **antes** de olhar o resultado. O texto explica
   o que você deveria esperar; a graça está em comparar com o que aconteceu.

### Três perguntas para levar ao final da aula

Quando um fornecedor apresentar um produto de IA no seu serviço, você deveria
conseguir fazer estas três perguntas — e sair daqui entendendo por que elas
importam:

1. **Em que população este modelo foi validado, e qual era a prevalência lá?**
2. **Qual é o ponto de corte, e quem o escolheu — vocês ou o fabricante?**
3. **O que acontece com o desempenho nos meus equipamentos e nos meus protocolos?**
"""
    )

    with st.expander("Sobre os modelos usados nesta aula"):
        for pesos, dados in MODELOS.items():
            st.markdown(
                f"**{dados['rotulo']}** — `{pesos}`  \n"
                f"Base: {dados['base']} · Origem: {dados['origem']}  \n"
                f"{dados['observacao']}"
            )
            st.markdown("")


# ------------------------------------------------------------------------------
# MÓDULO 1 — O que o modelo realmente vê
# ------------------------------------------------------------------------------
def pagina_preprocessamento(imagem, info, recortada, reduzida):
    cabecalho(
        "1 · O que o modelo realmente vê",
        "Antes de discutir acurácia, olhe para a imagem que a rede recebe.",
    )

    st.markdown(
        """
A radiografia que você abre no PACS tem tipicamente **2.000 × 2.500 pixels** e
12 a 16 bits de profundidade. A que a rede recebe tem **224 × 224 pixels** e
um canal. Entre uma e outra acontecem três coisas, e todas as três podem
custar informação clínica.
"""
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            para_exibir(imagem),
            caption=f"1. Como veio do arquivo — {imagem.shape[2]} × {imagem.shape[1]}",
            use_container_width=True,
        )
        st.caption(
            f"**Formato:** {info['formato']}  \n"
            f"**Profundidade:** {info['profundidade']}  \n"
            f"**Janela:** {info['janela']}"
        )

    with col2:
        st.image(
            para_exibir(recortada),
            caption=f"2. Recorte quadrado central — {recortada.shape[2]} × {recortada.shape[1]}",
            use_container_width=True,
        )
        perdido = 100 * (1 - (recortada.shape[1] * recortada.shape[2]) / (imagem.shape[1] * imagem.shape[2]))
        st.caption(
            f"A rede só aceita imagem quadrada. Este recorte descartou "
            f"**{num(perdido, 0)}%** da área — sempre das bordas, que é onde ficam "
            f"ápices e seios costofrênicos."
        )

    with col3:
        st.image(
            para_exibir(reduzida),
            caption=f"3. Reduzida — {reduzida.shape[2]} × {reduzida.shape[1]}",
            use_container_width=True,
        )
        pixels_originais = imagem.shape[1] * imagem.shape[2]
        pixels_finais = reduzida.shape[1] * reduzida.shape[2]
        st.caption(
            f"De **{num(pixels_originais, 0)}** para **{num(pixels_finais, 0)}** pixels. "
            f"Sobrou **{num(100 * pixels_finais / pixels_originais, 1)}%** da informação espacial."
        )

    st.divider()

    col_esq, col_dir = st.columns([1, 1])

    with col_esq:
        st.subheader("A quarta transformação, invisível")
        st.markdown(
            f"""
Além de recortar e reduzir, o valor de cada pixel é reescalado para a faixa
**[-1024, +1024]**, que é a escala que estes modelos esperam.

Nesta imagem, os valores vão de **{num(imagem.min(), 0)}** a **{num(imagem.max(), 0)}**.

Este é o erro silencioso mais comum de quem usa a biblioteca pela primeira vez:
passar uma imagem em [0, 255] direto para a rede. **Ela não dá erro.** Ela
devolve números com aparência perfeitamente normal, e completamente errados.
"""
        )

    with col_dir:
        figura, eixo = plt.subplots(figsize=(5.2, 3.2))
        eixo.hist(reduzida.ravel(), bins=60, color=COR_DESTAQUE, alpha=0.85)
        eixo.set_xlabel("Valor do pixel na escala do modelo")
        eixo.set_ylabel("Número de pixels")
        eixo.set_title("Distribuição dos valores após o pré-processamento", fontsize=10)
        eixo.spines[["top", "right"]].set_visible(False)
        figura.tight_layout()
        st.pyplot(figura)
        plt.close(figura)

    st.success(
        "**Para levar:** um nódulo de 4 mm numa radiografia de 2.500 pixels ocupa "
        "menos de um pixel depois da redução para 224 × 224. Estes modelos são "
        "razoáveis para achados grandes e difusos — derrame, cardiomegalia, "
        "consolidação extensa — e estruturalmente ruins para achados finos. Isso "
        "não é um defeito de treino: é uma consequência da resolução de entrada.",
        icon="🎯",
    )


# ------------------------------------------------------------------------------
# MÓDULO 2 — Lendo a saída do modelo
# ------------------------------------------------------------------------------
def pagina_inferencia(reduzida):
    cabecalho(
        "2 · Lendo a saída do modelo",
        "O número que aparece na tela não é o que você acha que é.",
    )

    pesos = st.selectbox(
        "Modelo",
        list(MODELOS.keys()),
        index=list(MODELOS.keys()).index(MODELO_PADRAO),
        format_func=lambda p: MODELOS[p]["rotulo"],
    )
    st.caption(MODELOS[pesos]["observacao"])

    with st.spinner("Rodando a rede… (na primeira vez, baixando os pesos)"):
        tabela = inferir_cache(pesos, reduzida)

    if tabela.empty:
        st.error("Este modelo não retornou nenhum achado válido.")
        return

    limiar = st.slider(
        "Ponto de corte para chamar de positivo",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help=(
            "0,50 é o ponto de operação escolhido pelos autores do modelo na "
            "validação. Arraste e veja quantos achados entram e saem."
        ),
    )

    col_grafico, col_tabela = st.columns([1.15, 1])

    with col_grafico:
        figura = grafico_barras(tabela, limiar)
        st.pyplot(figura)
        plt.close(figura)

    with col_tabela:
        acima = tabela[tabela["saida"] >= limiar]
        st.metric("Achados acima do corte", f"{len(acima)} de {len(tabela)}")

        exibicao = tabela.copy()
        exibicao["Saída"] = exibicao["saida"].map(lambda v: num(v, 3))
        exibicao["Sigmoide crua"] = exibicao["sigmoide"].map(lambda v: num(v, 4))
        exibicao["Limiar interno"] = exibicao["limiar"].map(lambda v: num(v, 4))
        st.dataframe(
            exibicao[["achado", "Saída", "Sigmoide crua", "Limiar interno"]].rename(
                columns={"achado": "Achado"}
            ),
            hide_index=True,
            use_container_width=True,
            height=min(430, 38 * len(exibicao) + 40),
        )

    st.divider()

    if not acima.empty:
        principal = acima.iloc[0]
        st.subheader(f"Achado mais pontuado: {principal['achado']}")
        st.caption(explicacao(principal["achado_en"]))

    with st.expander("Por que 0,73 NÃO quer dizer 73% de chance de ter a doença", expanded=True):
        exemplo = tabela.iloc[0]
        st.markdown(
            f"""
Pegue o achado do topo desta lista, **{exemplo['achado']}**, e compare os dois números:

| | valor | o que é |
|---|---|---|
| Sigmoide crua | `{num(exemplo['sigmoide'], 4)}` | a saída direta da rede, antes de qualquer ajuste |
| Limiar interno | `{num(exemplo['limiar'], 4)}` | o ponto de corte que os autores escolheram na validação |
| **Saída exibida** | **`{num(exemplo['saida'], 3)}`** | a sigmoide reescalada para que o limiar caia exatamente em 0,50 |

Repare no tamanho da sigmoide crua. Ela costuma ser pequena porque as doenças
são raras nas bases de treino — a rede aprende que "quase tudo é negativo" e
puxa todas as saídas para baixo. Se você lesse a sigmoide crua como
probabilidade, concluiria que quase nenhum exame tem nada.

A reescala (`op_norm`, na biblioteca) resolve o problema de leitura, mas cria
outro: o número **parece** uma probabilidade e não é. A tradução correta é:

- `saída > 0,50` → acima do ponto de corte do modelo
- `saída < 0,50` → abaixo do ponto de corte
- `saída = 0,73` → **bem** acima do corte, e nada mais que isso

Para transformar isso numa probabilidade de doença de verdade você precisa de
mais duas informações — sensibilidade/especificidade e prevalência. É
exatamente o assunto do módulo 3.
"""
        )

    with st.expander("O que são os 18 achados"):
        for _, linha in tabela.iterrows():
            st.markdown(f"**{linha['achado']}** ({linha['achado_en']}) — {explicacao(linha['achado_en'])}")


# ------------------------------------------------------------------------------
# MÓDULO 3 — Limiar, Bayes e valor preditivo
# ------------------------------------------------------------------------------
def pagina_bayes():
    cabecalho(
        "3 · Limiar, Bayes e valor preditivo",
        "O módulo mais importante desta aula — e o que menos tem a ver com programação.",
    )

    st.markdown(
        """
Um modelo de IA é um teste diagnóstico, e vale para ele tudo o que você já
aprendeu sobre testes diagnósticos.

Sensibilidade e especificidade são propriedades do teste: não mudam com a
população. Já o **valor preditivo positivo** — a chance de a pessoa realmente
ter a doença quando o teste dá positivo — depende da prevalência. E é o VPP,
não a sensibilidade, que determina o que você faz com o paciente à sua frente.

Mexa nos controles abaixo e observe o gráfico.
"""
    )

    col_controles, col_grafico = st.columns([1, 1.5])

    with col_controles:
        sensibilidade = st.slider("Sensibilidade do modelo", 0.50, 0.999, 0.90, 0.005)
        especificidade = st.slider("Especificidade do modelo", 0.50, 0.999, 0.90, 0.005)
        prevalencia = st.slider(
            "Prevalência do achado na população testada (%)",
            0.1, 60.0, 5.0, 0.1,
            help="Ou, para um paciente específico: a sua probabilidade pré-teste.",
        ) / 100.0

        rv_positiva, rv_negativa = razoes_de_verossimilhanca(sensibilidade, especificidade)
        vpp, doenca_apesar_negativo = probabilidade_pos_teste(
            prevalencia, sensibilidade, especificidade
        )

        st.metric("VPP — positivo do modelo", f"{num(vpp * 100, 1)}%")
        st.metric("VPN — negativo do modelo", f"{num((1 - doenca_apesar_negativo) * 100, 1)}%")
        st.metric("Razão de verossimilhança positiva", f"{num(rv_positiva, 1)}")
        st.metric("Razão de verossimilhança negativa", f"{num(rv_negativa)}")

    with col_grafico:
        figura = grafico_bayes(sensibilidade, especificidade, prevalencia)
        st.pyplot(figura)
        plt.close(figura)

    st.divider()

    # Cálculo concreto, em números absolutos — que é como as pessoas entendem.
    total = 10_000
    doentes = total * prevalencia
    sadios = total - doentes
    verdadeiros_positivos = doentes * sensibilidade
    falsos_positivos = sadios * (1 - especificidade)
    falsos_negativos = doentes * (1 - sensibilidade)
    verdadeiros_negativos = sadios * especificidade

    st.subheader("A mesma conta, em pessoas")
    st.markdown(
        f"""
Imagine **10.000 exames** passando por este modelo, numa população com
prevalência de **{num(prevalencia * 100, 1)}%**:
"""
    )

    tabela_2x2 = pd.DataFrame(
        {
            "Tem o achado": [num(verdadeiros_positivos, 0), num(falsos_negativos, 0)],
            "Não tem o achado": [num(falsos_positivos, 0), num(verdadeiros_negativos, 0)],
        },
        index=["Modelo apontou positivo", "Modelo apontou negativo"],
    )
    st.table(tabela_2x2)

    st.markdown(
        f"""
De cada **{num(verdadeiros_positivos + falsos_positivos, 0)}** alarmes que este
modelo dispara, apenas **{num(verdadeiros_positivos, 0)}** são pessoas que
realmente têm o achado. Os outros **{num(falsos_positivos, 0)}** vão gerar
tomografias, consultas, ansiedade e custo — sem doença nenhuma no fim.

E ainda escapam **{num(falsos_negativos, 0)}** pessoas que têm o achado e receberam
um negativo.
"""
    )

    st.warning(
        "**Para levar:** quando um artigo ou um fornecedor anunciar "
        "'acurácia de 95%', a pergunta certa não é se o número é verdadeiro. É: "
        "*medido em que população?* Mova o controle de prevalência entre 25% "
        "(pronto-socorro) e 0,5% (rastreamento) sem mexer em mais nada, e veja o "
        "VPP desabar. O modelo é o mesmo. Muda só quem entra na fila.",
        icon="🎯",
    )


# ------------------------------------------------------------------------------
# MÓDULO 4 — Onde o modelo olhou
# ------------------------------------------------------------------------------
def pagina_explicabilidade(reduzida):
    cabecalho(
        "4 · Onde o modelo olhou",
        "Mapa de ativação de classe: auditoria, não certificado de qualidade.",
    )

    pesos = st.selectbox(
        "Modelo",
        list(MODELOS.keys()),
        index=list(MODELOS.keys()).index(MODELO_PADRAO),
        format_func=lambda p: MODELOS[p]["rotulo"],
        key="modelo_cam",
    )

    modelo = carregar_modelo(pesos)
    tabela = inferir_cache(pesos, reduzida)

    if tabela.empty:
        st.error("Este modelo não retornou nenhum achado válido.")
        return

    escolha = st.selectbox(
        "Achado a explicar",
        tabela["achado_en"].tolist(),
        format_func=lambda a: f"{nome_pt(a)} — saída {num(tabela.loc[tabela.achado_en == a, 'saida'].iloc[0], 3)}",
    )

    indice = modelo.pathologies.index(escolha)
    mapa, logito, conferencia = mapa_de_ativacao(modelo, reduzida, indice)

    figura = figura_sobreposicao(reduzida, mapa, f"Contribuição por região — {nome_pt(escolha)}")
    st.pyplot(figura)
    plt.close(figura)

    st.caption(explicacao(escolha))

    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander("A conta fecha? (prova numérica)", expanded=False):
            st.markdown(
                f"""
Se o mapa de calor realmente decompõe a decisão do modelo, então a **média do
mapa somada ao viés** tem que dar exatamente o **logito** que a rede calculou
pelo caminho normal. Vamos conferir nesta imagem, para *{nome_pt(escolha)}*:

| | valor |
|---|---|
| Logito pelo caminho normal | `{num(logito, 6)}` |
| Média do mapa + viés | `{num(conferencia, 6)}` |
| Diferença | `{abs(logito - conferencia):.2e}` |

A diferença é da ordem do erro de arredondamento do computador. Ou seja: para
esta arquitetura, o mapa **não é uma aproximação** — é a decomposição exata da
conta, região por região.

Isso vale porque o DenseNet termina fazendo uma média espacial simples antes da
camada final. Em arquiteturas que não terminam assim, o mapa de calor passa a ser
estimado (Grad-CAM e parentes), e aí sim vira aproximação.
"""
            )

    with col_b:
        with st.expander("Como ler o mapa", expanded=False):
            st.markdown(
                """
- **Vermelho** — região que empurrou a conta **a favor** do achado
- **Azul** — região que empurrou **contra**
- **Neutro** — região que não pesou

Cada quadradinho do mapa original cobre 32 × 32 pixels da imagem de entrada
(o mapa nasce com 7 × 7 e é ampliado para 224 × 224). Em um tórax adulto, isso
é algo em torno de **5 cm**. A explicação é grosseira por construção: ela aponta
uma região, nunca uma estrutura.
"""
            )

    st.warning(
        "**Para levar:** um mapa que cai sobre o pulmão certo não prova que o "
        "modelo raciocinou como um radiologista. Ele mostra apenas quais pixels "
        "entraram na conta. Modelos que aprenderam atalhos — um dreno de tórax, "
        "eletrodos de monitorização, a marca de um aparelho portátil no canto da "
        "imagem — costumam produzir mapas de aparência perfeitamente plausível. "
        "Explicabilidade serve para você encontrar erros, não para confirmar acertos.",
        icon="🎯",
    )


# ------------------------------------------------------------------------------
# MÓDULO 5 — Modelos discordam entre si
# ------------------------------------------------------------------------------
def grafico_comparacao(tabela: pd.DataFrame, rotulos_colunas):
    """Mapa de calor achado × modelo, com os valores escritos em cada célula."""
    dados = tabela.to_numpy(dtype=float)

    figura, eixo = plt.subplots(
        figsize=(1.9 * dados.shape[1] + 2.5, 0.42 * dados.shape[0] + 2.0)
    )
    eixo.set_facecolor("#e8e8e8")  # cor das células sem valor

    desenho = eixo.imshow(
        np.ma.masked_invalid(dados), cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto"
    )

    eixo.set_xticks(range(dados.shape[1]))
    eixo.set_xticklabels(rotulos_colunas, rotation=30, ha="right", fontsize=8.5)
    eixo.set_yticks(range(dados.shape[0]))
    eixo.set_yticklabels(tabela.index, fontsize=9)

    for linha in range(dados.shape[0]):
        for coluna in range(dados.shape[1]):
            valor = dados[linha, coluna]
            texto = "—" if np.isnan(valor) else num(valor)
            cor = "#333333" if (np.isnan(valor) or 0.3 < valor < 0.7) else "white"
            eixo.text(coluna, linha, texto, ha="center", va="center", fontsize=8.5, color=cor)

    barra = figura.colorbar(desenho, ax=eixo, fraction=0.03, pad=0.02)
    barra.set_label("saída do modelo (0,50 = ponto de corte)", fontsize=8)
    barra.ax.tick_params(labelsize=7)

    eixo.set_title("Mesma radiografia, modelos treinados em populações diferentes", fontsize=10)
    figura.tight_layout()
    return figura


def pagina_discordancia(reduzida):
    cabecalho(
        "5 · Modelos discordam entre si",
        "Mesma imagem, mesma arquitetura, bases de treino diferentes.",
    )

    st.markdown(
        """
Todos os modelos abaixo são **DenseNet-121 de 224 × 224**. Mesma arquitetura,
mesmo número de parâmetros, mesmo código. A única diferença entre eles é em que
população foram treinados.

Se a inteligência estivesse na arquitetura, eles concordariam. Escolha pelo
menos três e veja o que acontece.
"""
    )

    escolhidos = st.multiselect(
        "Modelos a comparar (cada um novo baixa cerca de 30 MB na primeira vez)",
        list(MODELOS.keys()),
        default=["densenet121-res224-all", "densenet121-res224-nih", "densenet121-res224-pc"],
        format_func=lambda p: MODELOS[p]["rotulo"],
    )

    if len(escolhidos) < 2:
        st.info("Selecione pelo menos dois modelos para comparar.")
        return

    colunas = {}
    with st.spinner("Rodando cada modelo…"):
        for pesos in escolhidos:
            resultado = inferir_cache(pesos, reduzida)
            colunas[pesos] = resultado.set_index("achado_en")["saida"]

    comparacao = pd.DataFrame(colunas)

    # Divergência = maior diferença entre modelos para o mesmo achado. Calculada
    # ANTES de acrescentar qualquer coluna nova, senão ela entraria na conta.
    divergencia = comparacao.max(axis=1) - comparacao.min(axis=1)
    comparacao = comparacao.loc[divergencia.sort_values(ascending=False).index]
    comparacao.index = [nome_pt(achado) for achado in comparacao.index]

    figura = grafico_comparacao(comparacao, [MODELOS[p]["rotulo"] for p in escolhidos])
    st.pyplot(figura)
    plt.close(figura)

    st.caption(
        "Células cinzas com um traço: aquele modelo não foi treinado para aquele "
        "achado. Um modelo de tarefa única, como o da RSNA, tem quase tudo cinza."
    )

    maior = divergencia.sort_values(ascending=False)
    maior = maior[maior.notna()]
    if not maior.empty:
        achado_topo = maior.index[0]
        st.metric(
            f"Maior divergência: {nome_pt(achado_topo)}",
            f"{num(maior.iloc[0])}",
            help="Diferença entre a maior e a menor saída entre os modelos selecionados.",
        )

    st.divider()

    with st.expander("O experimento mais limpo: MIMIC-CheXpert contra MIMIC-NegBio"):
        st.markdown(
            """
Selecione, lá em cima, **apenas** estes dois modelos:

- MIMIC-CXR (rotulador CheXpert)
- MIMIC-CXR (rotulador NegBio)

Eles foram treinados nas **mesmas imagens**, dos **mesmos pacientes**, no
**mesmo hospital**, com a **mesma arquitetura**. A única diferença no mundo
entre os dois é qual programa leu os laudos de texto para produzir os rótulos
de treino.

Qualquer discordância que sobrar entre eles não vem da imagem, nem da doença,
nem da população. Vem inteiramente de como a palavra do radiologista foi
convertida em rótulo. É uma boa medida de quanto do "conhecimento" desses
modelos é, na verdade, ruído de anotação.
"""
        )

    st.warning(
        "**Para levar:** não existe 'a IA'. Existem modelos específicos, "
        "treinados em populações específicas. A pergunta que você deve fazer "
        "quando alguém apresentar um produto não é *funciona?*, é *funciona em "
        "quem?* — e, logo em seguida, *foi testado nos meus pacientes e nos meus "
        "equipamentos?*",
        icon="🎯",
    )


# ------------------------------------------------------------------------------
# MÓDULO 6 — Teste de robustez
# ------------------------------------------------------------------------------
def pagina_robustez(recortada):
    cabecalho(
        "6 · Teste de robustez",
        "Mudanças que não deveriam mudar nada — e o que o modelo faz com elas.",
    )

    st.markdown(
        """
Um radiologista que recebe a radiografia girada 15 graus, ou um pouco mais
escura, dá o mesmo laudo. Vamos verificar se o modelo também dá.

Cada perturbação abaixo é clinicamente irrelevante: nenhuma delas cria ou apaga
doença. Se a saída mudar muito, isso é uma **fragilidade do modelo**, e um
argumento concreto para exigir teste local antes de qualquer implantação.
"""
    )

    pesos = st.selectbox(
        "Modelo",
        list(MODELOS.keys()),
        index=list(MODELOS.keys()).index(MODELO_PADRAO),
        format_func=lambda p: MODELOS[p]["rotulo"],
        key="modelo_robustez",
    )

    referencia = inferir_cache(pesos, redimensionar(recortada))
    if referencia.empty:
        st.error("Este modelo não retornou nenhum achado válido.")
        return

    achado = st.selectbox(
        "Achado a acompanhar",
        referencia["achado_en"].tolist(),
        format_func=nome_pt,
    )

    linhas = []
    imagens = {}
    valor_base = None

    with st.spinner("Aplicando as perturbações…"):
        for rotulo, (tipo, intensidade) in PERTURBACOES.items():
            # A perturbação é aplicada ANTES da redução para 224, que é como
            # aconteceria na vida real (a imagem chega torta do aparelho).
            alterada = perturbar(recortada, tipo, intensidade)
            resultado = inferir_cache(pesos, redimensionar(alterada))

            selecao = resultado.loc[resultado["achado_en"] == achado, "saida"]
            valor = float(selecao.iloc[0]) if not selecao.empty else float("nan")

            if tipo == "nenhuma":
                valor_base = valor

            imagens[rotulo] = alterada
            linhas.append({"Perturbação": rotulo, "Saída": valor})

    tabela = pd.DataFrame(linhas)
    tabela["Variação"] = tabela["Saída"] - valor_base

    col_tabela, col_grafico = st.columns([1, 1.1])

    with col_tabela:
        exibicao = tabela.copy()
        exibicao["Saída"] = exibicao["Saída"].map(lambda v: num(v, 3))
        exibicao["Variação"] = exibicao["Variação"].map(
            lambda v: "—" if abs(v) < 1e-9 else num(v, 3, sinal=True)
        )
        st.dataframe(exibicao, hide_index=True, use_container_width=True)

        maior_variacao = tabela["Variação"].abs().max()
        st.metric(
            f"Maior variação em {nome_pt(achado)}",
            f"{num(maior_variacao, 3, sinal=True)}",
            help="Diferença absoluta máxima em relação à imagem original.",
        )
        if maior_variacao >= 0.15:
            st.error(
                "Variação grande. Alguma dessas transformações, que não muda nada "
                "clinicamente, mudou substancialmente a resposta do modelo.",
                icon="⚠️",
            )

    with col_grafico:
        figura, eixo = plt.subplots(figsize=(6, 0.45 * len(tabela) + 1.4))
        cores = [COR_ABAIXO if abs(v) < 1e-9 else (COR_ACIMA if abs(v) >= 0.10 else COR_DESTAQUE)
                 for v in tabela["Variação"]]
        eixo.barh(tabela["Perturbação"], tabela["Variação"], color=cores)
        eixo.axvline(0, color="black", linewidth=1)
        eixo.set_xlabel(f"Variação na saída para {nome_pt(achado)}")
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.tick_params(labelsize=8)
        figura.tight_layout()
        st.pyplot(figura)
        plt.close(figura)

    st.divider()
    st.subheader("As imagens que o modelo recebeu")

    colunas = st.columns(4)
    for posicao, (rotulo, matriz) in enumerate(imagens.items()):
        with colunas[posicao % 4]:
            st.image(
                para_exibir(redimensionar(matriz)),
                caption=rotulo,
                use_container_width=True,
            )

    st.warning(
        "**Para levar:** repare especialmente no **espelhamento horizontal**. "
        "Ele troca direita e esquerda — para um humano, a imagem espelhada de um "
        "coração normal vira uma dextrocardia. Se a saída do modelo quase não "
        "mudar, ele está ignorando lateralidade. Se mudar muito para um achado "
        "sem lado definido, ele está usando pistas que não deveriam importar. "
        "Nos dois casos você aprendeu algo que a AUC publicada não te contaria.",
        icon="🎯",
    )


# ------------------------------------------------------------------------------
# MÓDULO 7 — Viés: o que a rede aprendeu sem querer
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def carregar_modelo_idade():
    """Modelo do RIKEN que estima idade a partir da radiografia (erro médio ~3,7 anos)."""
    return _construir_modelo(
        xrv.baseline_models.riken.AgeModel,
        "o modelo de estimativa de idade (cerca de 440 MB)",
    )


@st.cache_resource(show_spinner=False)
def carregar_modelo_etnia():
    """Modelo do Emory HITI, replicando o achado de Gichoya et al. (2022)."""
    return _construir_modelo(
        xrv.baseline_models.emory_hiti.RaceModel,
        "o modelo do experimento de Gichoya (cerca de 85 MB)",
    )


def pagina_vies(recortada):
    cabecalho(
        "7 · Viés: o que a rede aprendeu sem querer",
        "A demonstração mais desconfortável desta aula — e a mais importante.",
    )

    st.markdown(
        """
Até aqui você viu modelos treinados para achar doença. Agora vamos ver o que
mais está gravado numa radiografia de tórax, sem que ninguém tenha pedido.

O ponto desta seção **não é** estimar idade ou etnia de alguém. É demonstrar,
com um exemplo que você pode rodar agora, que uma rede neural extrai da imagem
sinais que nós, humanos, não conseguimos sequer identificar. E, se ela consegue
extrair esses sinais, ela também pode estar **usando** esses sinais quando a
pergunta era outra — por exemplo, quando a pergunta era "tem pneumonia?".

É assim que nasce o viés algorítmico: não por alguém programar preconceito, mas
porque o modelo encontra atalhos correlacionados com características do
paciente e do serviço onde ele foi atendido.
"""
    )

    entrada = redimensionar(recortada, 320)

    aba_idade, aba_etnia = st.tabs(["Estimativa de idade", "O experimento de Gichoya et al."])

    # -- Idade ----------------------------------------------------------------
    with aba_idade:
        st.markdown(
            """
Este modelo, publicado por um grupo do RIKEN (Japão), estima a **idade
cronológica** a partir da radiografia de tórax, com erro médio absoluto em torno
de **3,7 anos**.

Pense no que isso significa: a informação "esta pessoa tem 58 anos" está
codificada nos pixels, de forma recuperável, mesmo que nenhum radiologista
consiga cravar a idade olhando o exame.
"""
        )

        if st.button("Estimar a idade a partir desta radiografia", key="botao_idade"):
            try:
                with st.spinner("Baixando e rodando o modelo de idade…"):
                    modelo = carregar_modelo_idade()
                    with torch.no_grad():
                        estimativa = float(modelo(para_tensor(entrada))[0, 0])

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric("Idade estimada", f"{num(estimativa, 0)} anos")
                with col_b:
                    st.caption(
                        "Erro médio absoluto reportado na publicação original: cerca "
                        "de 3,7 anos. Se você souber a idade real deste exame, "
                        "compare — mas lembre que uma acurácia alta aqui é justamente "
                        "o que deveria te preocupar, e não o contrário."
                    )
            except Exception as erro:
                st.error(f"Não consegui carregar o modelo de idade: {erro}")

        st.info(
            "**Por que isso importa clinicamente?** Um modelo treinado para detectar "
            "pneumonia numa base onde os pacientes mais velhos tinham mais pneumonia "
            "pode aprender a estimar idade e usar isso como atalho. Ele vai parecer "
            "excelente na validação — e vai errar sistematicamente quando a relação "
            "entre idade e doença for diferente na sua população.",
            icon="💡",
        )

    # -- Etnia ----------------------------------------------------------------
    with aba_etnia:
        st.markdown(
            """
Em 2022, Gichoya e colaboradores publicaram no *Lancet Digital Health* um
resultado que causou desconforto na área: redes neurais preveem a **etnia
autodeclarada** do paciente a partir de imagens médicas com desempenho alto —
e continuam conseguindo mesmo quando a imagem é borrada, cortada ou degradada
a ponto de nenhum radiologista reconhecer nada.

Até hoje não se sabe qual é o sinal que a rede usa. Não é densidade óssea, não
é índice de massa corporal, não é nenhuma das explicações testadas pelos autores.

**Leia o enquadramento antes de rodar.**
"""
        )

        st.error(
            "**O que este modelo NÃO é.** Etnia, aqui, é uma categoria social "
            "*autodeclarada pelo paciente* no registro hospitalar — não é uma "
            "variável biológica, e o modelo não está detectando nada de "
            "\"racial\" no corpo de ninguém. Este modelo não deve ser usado para "
            "inferir a etnia de nenhuma pessoa, em nenhuma circunstância. Ele está "
            "aqui por um único motivo: provar que a informação vaza para dentro da "
            "imagem, e que portanto ela pode ser usada como atalho por qualquer "
            "outro modelo treinado nessas mesmas bases.",
            icon="🛑",
        )

        if st.button("Rodar o experimento", key="botao_etnia"):
            try:
                with st.spinner("Baixando e rodando o modelo…"):
                    modelo = carregar_modelo_etnia()
                    with torch.no_grad():
                        saida = modelo(para_tensor(entrada))
                        # A rede devolve logitos; a softmax os converte em
                        # proporções que somam 1.
                        proporcoes = torch.softmax(saida, dim=1)[0].numpy()

                traducao = {"Asian": "Asiática", "Black": "Negra", "White": "Branca"}
                tabela = pd.DataFrame(
                    {
                        "Categoria (autodeclarada)": [traducao.get(a, a) for a in modelo.targets],
                        "Proporção atribuída": [f"{num(v * 100, 1)}%" for v in proporcoes],
                    }
                )
                st.dataframe(tabela, hide_index=True, use_container_width=True)

                st.caption(
                    "O ponto do experimento não é se o modelo acertou. É que ele "
                    "consegue ser confiante — a partir de uma imagem em tons de "
                    "cinza de 224 × 224 pixels, onde nenhum radiologista arriscaria "
                    "palpite nenhum."
                )
            except Exception as erro:
                st.error(f"Não consegui carregar o modelo: {erro}")

        st.warning(
            "**Para levar:** o desdobramento prático desse achado veio em outro "
            "estudo (Seyyed-Kalantari et al., *Nature Medicine*, 2021): modelos de "
            "radiografia de tórax **subdiagnosticam** de forma sistemática pacientes "
            "de grupos historicamente menos assistidos — deixam de sinalizar doença "
            "justamente em quem tem menos acesso a uma segunda opinião. O modelo não "
            "foi programado para isso. Ele apenas aprendeu, dos dados, a reproduzir "
            "as desigualdades que estavam nos dados. Auditar desempenho por subgrupo "
            "não é burocracia: é a única forma de enxergar esse tipo de falha.",
            icon="🎯",
        )


# ------------------------------------------------------------------------------
# MÓDULO 8 — Segmentação anatômica
# ------------------------------------------------------------------------------
# As 14 estruturas que o modelo de segmentação delimita.
# "Weasand" é uma palavra arcaica do inglês para esôfago — os autores da base
# usaram esse termo, e mantivemos a chave original para o código não quebrar.
ESTRUTURAS = {
    "Left Lung": "Pulmão esquerdo",
    "Right Lung": "Pulmão direito",
    "Heart": "Coração",
    "Aorta": "Aorta",
    "Mediastinum": "Mediastino",
    "Spine": "Coluna vertebral",
    "Facies Diaphragmatica": "Face diafragmática",
    "Left Clavicle": "Clavícula esquerda",
    "Right Clavicle": "Clavícula direita",
    "Left Scapula": "Escápula esquerda",
    "Right Scapula": "Escápula direita",
    "Left Hilus Pulmonis": "Hilo pulmonar esquerdo",
    "Right Hilus Pulmonis": "Hilo pulmonar direito",
    "Weasand": "Esôfago",
}


@st.cache_resource(show_spinner=False)
def carregar_modelo_segmentacao():
    """PSPNet treinada na base ChestX-Det para delimitar 14 estruturas do tórax."""
    return _construir_modelo(
        xrv.baseline_models.chestx_det.PSPNet,
        "o modelo de segmentação (cerca de 260 MB)",
    )


def _largura_maxima(mascara: np.ndarray) -> int:
    """Maior extensão horizontal de uma máscara binária, em pixels."""
    maior = 0
    for linha in mascara:
        indices = np.flatnonzero(linha)
        if indices.size:
            maior = max(maior, int(indices[-1] - indices[0] + 1))
    return maior


def pagina_segmentacao(recortada):
    cabecalho(
        "8 · Segmentação anatômica",
        "Classificar não é a única coisa que IA faz com imagem.",
    )

    st.markdown(
        """
Os módulos anteriores usaram modelos de **classificação**: entra uma imagem,
sai uma lista de probabilidades. Aqui a tarefa é outra — **segmentação**: para
cada pixel, o modelo decide a que estrutura ele pertence.

Na prática clínica, esse é o tipo de modelo que costuma ser mais útil e mais
fácil de auditar: em vez de um número opaco, ele produz um contorno que você
consegue conferir com os próprios olhos, e a partir do qual dá para calcular
medidas objetivas.
"""
    )

    escolhidas = st.multiselect(
        "Estruturas a desenhar",
        list(ESTRUTURAS.keys()),
        default=["Left Lung", "Right Lung", "Heart"],
        format_func=lambda e: ESTRUTURAS[e],
    )

    if not st.button("Rodar a segmentação", key="botao_segmentacao"):
        st.info("Clique no botão para baixar e rodar o modelo de segmentação.")
        return

    try:
        with st.spinner("Baixando e rodando a PSPNet…"):
            modelo = carregar_modelo_segmentacao()
            entrada = redimensionar(recortada, 512)
            with torch.no_grad():
                saida = modelo(para_tensor(entrada))
            # A saída traz um mapa por estrutura; a sigmoide converte cada pixel
            # numa "confiança" entre 0 e 1, e 0,5 é o corte usual.
            mapas = torch.sigmoid(saida)[0].numpy()  # [14, 512, 512]
    except Exception as erro:
        st.error(f"Não consegui carregar o modelo de segmentação: {erro}")
        return

    paleta = matplotlib.colormaps["tab10"]

    figura, eixo = plt.subplots(figsize=(7, 7))
    eixo.imshow(para_visualizacao(entrada), cmap="gray")

    for posicao, estrutura in enumerate(escolhidas):
        indice = modelo.targets.index(estrutura)
        eixo.contour(
            mapas[indice],
            levels=[0.5],
            colors=[paleta(posicao % 10)],
            linewidths=1.8,
        )
        eixo.plot([], [], color=paleta(posicao % 10), label=ESTRUTURAS[estrutura])

    eixo.legend(loc="lower right", fontsize=8, framealpha=0.85)
    eixo.axis("off")
    eixo.set_title("Contornos previstos pelo modelo", fontsize=11)
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

    st.divider()

    # -- Índice cardiotorácico -------------------------------------------------
    st.subheader("Uma medida clínica de verdade: o índice cardiotorácico")

    indice_coracao = modelo.targets.index("Heart")
    indice_esquerdo = modelo.targets.index("Left Lung")
    indice_direito = modelo.targets.index("Right Lung")

    mascara_coracao = mapas[indice_coracao] > 0.5
    mascara_torax = (mapas[indice_esquerdo] > 0.5) | (mapas[indice_direito] > 0.5)

    largura_coracao = _largura_maxima(mascara_coracao)
    largura_torax = _largura_maxima(mascara_torax)

    if largura_coracao == 0 or largura_torax == 0:
        st.warning(
            "O modelo não conseguiu delimitar o coração ou os pulmões nesta imagem. "
            "Isso costuma acontecer em exames muito recortados, em incidências "
            "atípicas ou em imagens com artefato importante — e, por si só, já é "
            "uma informação útil sobre os limites do modelo."
        )
        return

    ict = largura_coracao / largura_torax

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Largura cardíaca", f"{largura_coracao} px")
    col_b.metric("Largura torácica", f"{largura_torax} px")
    col_c.metric("Índice cardiotorácico", f"{num(ict)}")

    if ict > 0.5:
        st.markdown(
            f"Um ICT de **{num(ict)}** ficaria acima do corte clássico de 0,50, o que "
            "**sugeriria** cardiomegalia em uma incidência PA adequada."
        )
    else:
        st.markdown(
            f"Um ICT de **{num(ict)}** ficaria dentro do limite clássico de 0,50."
        )

    st.caption(
        "Ressalvas importantes: (1) o ICT clássico usa a margem interna dos arcos "
        "costais, e aqui estamos usando a extensão dos pulmões como aproximação; "
        "(2) o índice só é válido em PA com boa inspiração — em AP e em exames de "
        "leito o coração é magnificado; (3) tudo isso é medido sobre uma imagem "
        "recortada e reduzida a 512 pixels. É demonstração didática, não medição."
    )

    st.success(
        "**Para levar:** compare este número com a saída de *Cardiomegalia* no "
        "módulo 2, para a mesma radiografia. Quando os dois discordam, você tem um "
        "caso concreto para discutir: qual dos dois você conferiria primeiro? O "
        "modelo que devolve um número sem justificativa, ou o que devolve um "
        "contorno que você pode medir e contestar? Essa é, no fundo, a diferença "
        "entre uma ferramenta que você audita e uma que você apenas obedece.",
        icon="🎯",
    )


# ==============================================================================
# BLOCO 9 — O ROTEADOR
# ==============================================================================
# Decide qual módulo desenhar. Toda a preparação da imagem acontece uma vez só,
# aqui, e o resultado é passado adiante — assim os módulos ficam independentes e
# você pode reordenar, remover ou acrescentar um sem mexer nos outros.
# ==============================================================================


def main():
    modulo, caminho = barra_lateral()

    # A página inicial é a única que não precisa de imagem nenhuma.
    if modulo == MODULOS[0]:
        pagina_inicio()
        return

    if caminho is None:
        cabecalho(modulo)
        st.warning(
            "Escolha uma radiografia na barra lateral, à esquerda, para continuar.",
            icon="👈",
        )
        return

    try:
        imagem, info, recortada, reduzida = preparar(caminho)
    except Exception as erro:
        cabecalho(modulo)
        st.error(
            "Não consegui ler esta imagem.\n\n"
            f"Detalhe técnico: `{erro}`\n\n"
            "Se for um DICOM, confirme que o pydicom está instalado "
            "(`python -m pip install pydicom`). Se for um PNG ou JPG, confirme que "
            "o arquivo não está corrompido."
        )
        return

    # As falhas de carregamento de modelo viram RuntimeError com uma mensagem já
    # escrita para o aluno (veja _construir_modelo). Sem este try, o Streamlit
    # despejaria um traceback de 30 linhas na tela, escondendo a instrução útil.
    try:
        if modulo == MODULOS[1]:
            pagina_preprocessamento(imagem, info, recortada, reduzida)
        elif modulo == MODULOS[2]:
            pagina_inferencia(reduzida)
        elif modulo == MODULOS[3]:
            pagina_bayes()
        elif modulo == MODULOS[4]:
            pagina_explicabilidade(reduzida)
        elif modulo == MODULOS[5]:
            pagina_discordancia(reduzida)
        elif modulo == MODULOS[6]:
            pagina_robustez(recortada)
        elif modulo == MODULOS[7]:
            pagina_vies(recortada)
        elif modulo == MODULOS[8]:
            pagina_segmentacao(recortada)
    except RuntimeError as erro:
        st.error(str(erro), icon="⚠️")


# ==============================================================================
# BLOCO 10 — COMO ESTE ARQUIVO É EXECUTADO
# ==============================================================================
# Um detalhe do Streamlit que confunde muita gente: quando você roda
# `streamlit run arquivo.py`, o Python ainda assim define __name__ == "__main__".
# Ou seja, os dois modos de execução caem no mesmo lugar. Precisamos distinguir.
#
# Se o arquivo foi chamado com `python arquivo.py`, não existe servidor Streamlit
# rodando. Em vez de quebrar com uma mensagem incompreensível, imprimimos as
# instruções corretas — inclusive as células prontas para o Google Colab.
# ==============================================================================

INSTRUCOES = """
================================================================================
  Este arquivo é um aplicativo Streamlit. Ele não roda com `python`.
================================================================================

NO SEU COMPUTADOR
-----------------
  streamlit run aula_ia_radiologia.py

  O navegador abre sozinho em http://localhost:8501
  Para encerrar, volte ao terminal e pressione Ctrl+C.


NO GOOGLE COLAB
---------------
O Colab não mostra páginas do Streamlit direto no notebook: é preciso abrir um
túnel para fora. Cole cada bloco abaixo em uma célula separada, nesta ordem.

  # Célula 1 — instalar as dependências (leva 1 a 2 minutos)
  !pip install -q torchxrayvision pydicom streamlit
  !npm install -g localtunnel

  # Célula 2 — enviar este arquivo
  from google.colab import files
  files.upload()          # selecione aula_ia_radiologia.py

  # Célula 3 — mostrar a senha do túnel (anote: você vai precisar dela)
  !curl -s https://loca.lt/mytunnelpassword

  # Célula 4 — subir o aplicativo e abrir o túnel
  !streamlit run aula_ia_radiologia.py &>/content/registro.txt &
  !npx localtunnel --port 8501

Clique no endereço que aparecer na célula 4, cole a senha da célula 3 e pronto.
Se algo der errado, o que o Streamlit escreveu está em /content/registro.txt.


LEMBRETE
--------
Material didático. Não é dispositivo médico, não tem registro na ANVISA e não
pode ser usado para decisão clínica sobre nenhuma pessoa real. Não envie
imagens de pacientes identificáveis.
================================================================================
"""


def _rodando_no_streamlit() -> bool:
    """Descobre se existe um servidor Streamlit de verdade por trás deste código."""
    try:
        return st.runtime.exists()
    except Exception:
        # Caminho alternativo, para versões mais antigas da biblioteca.
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            return get_script_run_ctx() is not None
        except Exception:
            return False


if __name__ == "__main__":
    if _rodando_no_streamlit():
        main()
    else:
        # A codificação da saída já foi ajustada lá no BLOCO 0, então os acentos
        # abaixo saem corretos mesmo no terminal do Windows.
        print(INSTRUCOES)
