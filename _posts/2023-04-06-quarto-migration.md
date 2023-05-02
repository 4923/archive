---
toc: true
badges: true
comments: true

layout: post
keywords: fastai, quarto, migration
description: 나를 두고 떠나간 fastpages의 자취를 찾아서
categories: [migration, blog-operating]
permalink: /fastpages-2-quarto/
title: A migration log from Fastpages to Quarto
---

> 왜 이런 시도를 했는가
>> To Blogging only using `Jupyter Notebook` by rendering quarto APIs
- It doesn't need `nbdev` which is used to use fastpages or nbdev1. So those jekyll blogs should be migrated into quarto (Jupyter into Jupyter for quarto, Markdown into Jupyter-ish sth)
- Those are why the *previous fastpages* doesn't utilize CSS because new version of fastpages: Quarto is only consisted of Jupyter Notebook format.

### 발단

어느날부터 docker image로 돌리는 서버에서 CSS가 전부 깨지기 시작하는 문제가 발생했다. (확실히 관계있는 이슈인지는 알 수 없음)   
CSS 커스텀은 종종 해오던 일이라 이번에 뭔갈 잘못 건드린 줄 알고 따로 repo/issue를 찾아보지는 않았는데 아무래도 이상해서 확인해봤더니 deprecated 되었다고 한다.  

<img width="882" alt="fastpages deprecated announcement in fastpage README" src="https://user-images.githubusercontent.com/60145951/230151370-c4b6384a-40e3-43ee-b612-2b7f13ffda4e.png">

fastai를 믿었는데... 갑자기 이전됐다고 하길래 버티려고 했으나 주피터를 변환해주는 곳은 여기뿐일 것 같아서 얌전히 migration 하기로 했다.



##  installation
1. [Get Started](https://quarto.org/docs/get-started/) 에서 `Quarto`를 설치한다. 
    - 지원 환경 : Ubuntu 18+ /Debian 10+, Linux Arm64, Mac OS, Windows
2. **VSC**, Jupyter, RStudio, Neovim, Text Editor 등을 이용해 시작.
3. VSC의 경우 [Extension: quarto-vscode](https://marketplace.visualstudio.com/items?itemName=quarto.quarto) 을 설치해야 한다.

Windows, Linux Arm64, Ubuntu18+/Debian10+ 을 비롯해 Mac도 지원을 하므로 pkg 파일을 다운받을 수는 있으나 brew 또는 brew cask로 통일해서 관리하는걸 좋아해서 `brew install --cask quarto` 로 quarto를 설치했다. ([homebrew: quarto](https://formulae.brew.sh/cask/quarto#default))
- brew/quarto 에서는 quarto를 'Scientific and technical publishing system built on Pandoc' 라고 소개하고 있다.

<img width="731" alt="image" src="https://user-images.githubusercontent.com/60145951/230153234-38d1d734-fdff-404f-b0fa-49ca2a20c8e2.png">

- `--appdir` option은 brew의 옵션 중 하나로, 원래 /Applications 경로에 있어야 하는걸 다른 쪽으로 옮기는게 요구사항인가 지시사항인가 그랬을텐데 그게 지금 막혀있으니까 (ignore...) 관리자 권한으로 root (`/`) 에 옮기기 위해 PW를 요구한 것으로 보인다.

이후 각 프로세스에 따라 jupyter의 front-matter meta data를 수정하는 작업을 거치고 markdown은 qmd로 변환하는 작업을 거쳐야하는데 이 공수나 새로 프로젝트를 만들어 내용을 정돈하는 공수나 비슷할 것 같아 오류가 발생할 가능성이 높은 수동 이전migration이 아닌 새 프로젝트 생성을 선택했다.


### Create new project
> 전반적으로 메타데이터를 잘 활용할 수 있어야 한다. (잘) 구성된 HTML 레이아웃을 제공하고 그에 맞추어 렌더링되므로 적절한 메타데이터가 필요할 것.

> 효율적인 시간 관리를 위해 migration 대신 새 프로젝트를 생성했다.

![rendering workflow](https://quarto.org/docs/get-started/hello/images/qmd-how-it-works.png)


1. quarto 명령어로 새 프로젝트 생성

- 일단 최소한의 구조를 만들어 준다. fastpages는 레포 클론해서 PR날리는 등 번거로운 절차가 있는 반명 이건 cli 환경에서 바로 프로젝트를 생성해주므로 편한 것으로 보인다.
- migration 과정이 아닌 단순 프로젝트 생성에도 동일한 명령어를 사용하는데 이 때의 명령어는 : `quarto create-project {project name}` 이다.

```bash
$ quarto create-project --type website:blog .   // 기존 dir
$ quarto create-project {project name}          // 새 프로젝트
$ quarto install extension quarto-ext/video
```

### Jupyter Notebook (`.ipynb`)
jupyter는 거의 형식이 동일하므로 크게 바꿀만한 내용이 없다. 세부 설정은 `{project name}/_quarto.yaml` 에서 통일하며, 오버라이딩은 맨 위 마크다운셀에서 `---` 로 감싸 진행한다.

### Quick Rendering
```sh
$ quatro render {target project}
# or Using shortcut key `Ctrl + Shift + K` renders specific file
# in those case, control (ctrl) key in Windows matches cmd when it comes to MacOS
```
quarto는 Render 를 통해 라이브서버 또는 로컬호스트로 서버 돌리지 않아도 바로 볼 수 있다는 장점이 있다.

|<img width="1800" alt="image" src="https://user-images.githubusercontent.com/60145951/235649610-269d741e-9c0b-4186-a8a4-2f9585c81231.png">|
|:-:|
|좌측은 jupyter와 local live server, 우측은 실제로 publishing 한 github page다.|

`quarto`는 `/nbs` 에 포스트들을 모아두므로 해당 위치로 이동한 후 `quarto render`명령어를 사용하거나 `quarto render {path of nbs}` 명령어를 사용하면 프로젝트 전체가 렌더링된다.


## Web page project Searching

포스트를 만들면 그 밑으로 쭉 생성되는데 계층구조를 만들기 위해서는 따로 작업을 해야하는 것 같음. 당장 필요하진 않을거같음. 옮길 때 주의해서 옮기자.