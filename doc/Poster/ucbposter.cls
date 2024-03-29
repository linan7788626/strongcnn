%%
%%    File : ucbposter.cls    (LaTeX2e class file)
%%           Based on infposter.cls of the University of Edinburgh. 
%%             
%%  Author : Version 1.0 by Martin Reddy (mxr@dcs.ed.ac.uk) 
%%           Version 2.0 by Mary Ellen Foster (mef@cogsci.ed.ac.uk)
%% Updates : 1.0 [14/2/97] - initial release.
%%           1.1 [14/3/97] - center caption, colour changes
%%           2.0 [4/5/2000] - new layout, incorporating new informatics
%%                            logo.  (Renamed to ucbposter)
%%           2.1 [11/28/2000] - use new informatics seal, graphics rather
%%                            than graphicx.
%%	         2.2 [04/05/2001] - margins reduced to avoid blue frame
%%				disappearing on A0 printer
%%           2.3 [10/05/2002] - fixed the way sections get redefined; added \email
%               and \homepage commands.
%%           3.0 [21/11/2007] - Initial version adapted to Berkeley
%% 
%% This file contains a class definition, ucbposter, for the LaTeX2e
%% system which defines the layout of posters adapted from the style of
%% the University of Edinburgh.
%%
%% The following LaTeX packages are required by this class:
%%    pstricks - to embed the background EU crests and border.
%%    graphics - to display all EPS figures
%%    multicol - to produce the 3 column text.
%%    sectsty  - to redefine the font and colour of the headers
%%    url      - to typeset urls properly
%%

\NeedsTeXFormat{LaTeX2e}[1994/12/01]
\ProvidesClass{ucbposter}[2007/11/21 v3.0 UCB Poster Class, based on UoE infposter.cls]

%% tweaks to permit use with FoilTeX 2.1
\newif\if@magscaleECfonts \@magscaleECfontsfalse
\newif\if@useDCfonts \@useDCfontsfalse

%%
%% --- Initial Code ---
%%

\newlength{\posterwidth}
\newlength{\posterheight}
  \setlength\posterheight{840mm}
\setlength\posterwidth{1188mm}

% MEF added these
\newcommand{\poster@columns}{}
\newcommand{\poster@depturl}{\url{http://www.eecs.berkeley.edu}}
\newcommand{\poster@deptimage}{school_of_informatics.eps}
\newcommand{\poster@deptimageheight}{1.5in}
\newcommand{\poster@mainimage}{GraphicB-pale.eps}
\newcommand{\poster@deptname}{}
\newcommand{\poster@centre}{}

\newif\if@posterland
% Various possible layout configurations and their defaults (many of these
% cannot be set to otherwise, but I'll leave the code in just in case)
\newif\if@posteroverlap \@posteroverlaptrue
\newif\if@posterborder \@posterbordertrue
\newif\if@postercolumns \@postercolumnstrue
\newif\if@posterseal \@postersealtrue
\newif\if@posterunderseal \@posterundersealfalse

%%
%%  --- Options ---
%%


\DeclareOption{center}{\renewcommand{\poster@centre}{yes}}
\DeclareOption{centre}{\renewcommand{\poster@centre}{yes}}

%% Added one-column option
\DeclareOption{onecolumn}{\@postercolumnsfalse}
\DeclareOption{twocolumn}{\@postercolumnstrue\renewcommand{\poster@columns}2}
\DeclareOption{threecolumn}{\@postercolumnstrue\renewcommand{\poster@columns}3}
\DeclareOption{fourcolumn}{\@postercolumnstrue\renewcommand{\poster@columns}4}

\DeclareOption{landscape}{\@posterlandtrue}
\DeclareOption{portrait}{\@posterlandfalse}

% Should the poster have a border around it?
\DeclareOption{border}{%
    \@posterbordertrue
}
\DeclareOption{noborder}{%
    \@posterborderfalse
}

% Default options
\ExecuteOptions{threecolumn,landscape}
\ProcessOptions

%%
%% --- Class Loading (built ontop of article.cls) ---
%%

\LoadClass{article}

%%
%% --- Main Code ---
%%

\usepackage{pstricks}
% Epsfig retained for compatibility, but all graphics are included through
% graphicx
\usepackage{epsfig}
\usepackage{graphics}
\usepackage{multicol}
\usepackage{ucbseal}
\sealtype{0}
% Added this in case urls are to be displayed
\PassOptionsToPackage{obeyspaces}{url}
\usepackage{url}
% A better way to redefine section headers!
\usepackage{sectsty}

%% Set up the default colours for the poster.
%% These can be modified directly by the user

\let\ItemizeColour\magenta
\let\SectionColour\blue
\let\TitleColour\black
\let\SubtitleColour\black
\let\AuthorColour\black
\let\AbstractColour\black
\let\CaptionColour\red

%% provide a number of lengths to let the user
%% customise the spacing of the poster

\newlength{\SpaceAfterTitle}
\newlength{\SpaceAfterSubtitle}
\newlength{\SpaceAfterAuthor}
\newlength{\SpaceAfterAbstract}
\newlength{\SpaceBeforeText}
\newlength{\SpaceToColumnSep}
\newlength{\SpaceTemp}

\setlength{\SpaceToColumnSep}{5em}

\newenvironment{posterindent}
   {\list{}{\setlength\leftmargin\SpaceToColumnSep%
   \setlength\rightmargin\leftmargin}\item\relax}
   {\endlist}

%% keep a track of whether \makeposter has been called. If it has
%% then we need to do an \end{multicols}

\def\poster@true{1}
\def\poster@false{0}
\let\poster@displayed\poster@false

% End whatever environments we're in (the exact ones depend on the options)
\AtEndDocument{%
  \if\poster@displayed\poster@true \end{posterindent} 
    \if@postercolumns
      \end{multicols} 
    \fi%
  \fi%
}

%% setup the multicols environment

\columnseprule=.5mm
\flushcolumns

%% Setup some really big fonts. This is a pain to do, so I'm just
%% going to use the style files from the FoilTeX system which already
%% do this.

\input{fltfonts.def}
\input{foil30.clo}
\renewcommand\familydefault{\rmdefault}

% Modify the FoilTeX defs a bit so we can have *really* big font 
% sizes for the main title and sub title.

\DeclareFontShape{OT1}{fcmr}{bx}{n}{%
  <12><14.4><17.28><20.74><24.88><29.86><35.83><43.00><51.60><72>cmbx10}{}

\DeclareFontShape{OT1}{fcmr}{bx}{sl}{%
  <12><14.4><17.28><20.74><24.88><29.86><35.83><43.00><51.60><64>cmbxsl10}{}

\newcommand{\poster@maintitlefont}{%
  \fontencoding{OT1}\fontfamily{fcmr}\fontseries{bx}%
  \fontshape{n}\fontsize{72}{84}\selectfont%
}

\newcommand{\poster@subtitlefont}{%
  \fontencoding{OT1}\fontfamily{fcmr}\fontseries{bx}%
  \fontshape{sl}\fontsize{64}{72}\selectfont%
}

%% define the page size and various margins/offsets

\setlength{\topmargin}{0in}
\setlength{\oddsidemargin}{27.1mm}
\setlength{\evensidemargin}{27.1mm}

\setlength{\footskip}{0.5in}
\setlength{\headheight}{0in}
\setlength{\headsep}{0in}
\setlength{\voffset}{0in}
\setlength{\hoffset}{0in}

\if@posterland
  \typeout{Landscape Poster}
  \special{papersize=1188mm,840mm}
\else
  \typeout{Portrait Poster}
  \newlength\poster@tmpw
  \setlength\poster@tmpw\posterwidth
  \setlength\posterwidth\posterheight
  \setlength\posterheight\poster@tmpw
  \special{papersize=840mm,1188mm}
\fi


\setlength{\textwidth}{0.93\posterwidth}
\setlength{\textheight}{0.91\posterheight}

%% Define lengths for the poster border, seal position, etc.

\newlength{\borderheight}
\setlength{\borderheight}{\posterheight}
\addtolength{\borderheight}{-2.166in}

\newlength{\borderwidth}
\setlength{\borderwidth}{\posterwidth}
\addtolength{\borderwidth}{-2.5in}

\newlength{\sealoffset}
\setlength{\sealoffset}{\posterwidth}
\addtolength{\sealoffset}{-5.5in}

%% Add a \special for dvips to generate correct PS
%%   a4 210mm 297mm, a3 297mm 420mm, a2 420mm 594mm
%%   a1 594mm 840mm, a0 840mm 1188mm  (33.07in x 46.77in)


%% Specify the embedded PS layout to do at the start of the document
%% This includes the greyed-logo and colour seal in the top corners
%% and the blue frame around the poster.

\AtBeginDocument{%
  \if@posterseal
    \rput(\sealoffset,-2.74in){
      \resizebox{5in}{!}{\includeseal}
    }%
  \fi
%%  % Put in the logo
%%  \if@posterland
%%      \rput(5in,-15.5in){
%%        \resizebox{!}{\textheight}{\includegraphics{\poster@mainimage}}
%%      }%
%%  \else
%%      \rput(7in,-22in){
%%        \resizebox{!}{\textheight}{\includegraphics{\poster@mainimage}}
%%      }%
%%  \fi
  % If there's a border, draw it
  \if@posterborder
    \psframe[linewidth=1mm,linecolor=blue,framearc=0.004,fillstyle=none]%
    (-0.5in,-\borderheight)(\borderwidth,0.166in)%
  \else
  \fi
  % Now we will add some empty boxes in order to put TeX stuff beyond
  % all the pstricks stuff, otherwise it gets ignored by dvips -E
  % Notice: this causes "overfull hbox" errors, but no problems
  \hskip -1in\hbox{\white\vrule height0.7in depth0pt width1pt}%
  \setlength\SpaceTemp{\posterwidth}%
  \addtolength\SpaceTemp{-1in}%
  \hskip \SpaceTemp%
  \hbox{\white\vrule height0.7in depth0pt width1pt}%
}

%% Macros to input the poster titles, author, etc.

\newcommand{\poster@title}{}
\newcommand{\poster@subtitle}{}
\newcommand{\poster@author}{}
\newcommand{\poster@email}{}
\newcommand{\poster@url}{}
\newcommand{\poster@abstract}{}
\newcommand{\poster@empty}{}

\newcommand{\subtitle}[1]{\renewcommand{\poster@subtitle}{#1}}
\newcommand{\enquiries}[1]{\renewcommand{\poster@email}{#1}}
\newcommand{\email}[1]{\renewcommand{\poster@email}{\url{#1}}}
\newcommand{\homepage}[1]{\renewcommand{\poster@url}{\url{#1}}}
\renewcommand{\title}[1]{\renewcommand{\poster@title}{#1}}
\renewcommand{\author}[1]{\renewcommand{\poster@author}{#1}}
\renewcommand{\abstract}[1]{\renewcommand{\poster@abstract}{#1}}

\newlength{\titlewidth}
% Take into account overlapping (probably to be deleted)
\if@posteroverlap
  \setlength{\titlewidth}{.75\textwidth}
\else
  \setlength{\titlewidth}{\textwidth}
  \addtolength{\titlewidth}{-\poster@mainindent}
  \setlength{\titlewidth}{.75\titlewidth}
\fi

%% Define the \makeposter command which generates the poster header.
%% We also keep a track of whether this is called or not so that we
%% know if we are to do anything at the end of the document.

\newcommand{\makeposter}{%
  \if\poster@displayed\poster@false
    \let\poster@displayed\poster@true
    % Indent if we're not to overlap (probably to be deleted)
    \if@posteroverlap \else \begin{mainposterindent} \fi
    \begin{posterindent}
      % If we're supposed to skip a seal, do it (probably to be deleted)
      \if@posterunderseal
        \vspace*{7in}
      \fi
      \begin{minipage}{0pt}\vspace*{5in}\hfill\end{minipage}
      \begin{minipage}{\titlewidth}
        \ifx\poster@title\poster@empty \typeout{WARNING: No Title} \else
          \vskip -20pt{\TitleColour\poster@maintitlefont\poster@title}%
          \vskip \SpaceAfterTitle%
        \fi
        \ifx\poster@subtitle\poster@empty \else
          \vskip 20pt{\SubtitleColour\poster@subtitlefont\poster@subtitle}%
          \vskip \SpaceAfterSubtitle%
        \fi
        \ifx\poster@author\poster@empty \else
          \vskip 50pt{\AuthorColour\Huge\bfseries\em\poster@author}%
          \ifx\poster@email\poster@empty \else
              \vskip 20pt{\Large\poster@email}
          \fi
          \ifx\poster@url\poster@empty \else
              \vskip 15pt{\Large\poster@url}
          \fi
          \vskip \SpaceAfterAuthor%
        \fi
        \ifx\poster@abstract\poster@empty \else
          \vskip 50pt{\AbstractColour\large\bfseries\underbar{Abstract}:
          \poster@abstract}\vskip \SpaceAfterAbstract%
        \fi
        \advance\SpaceBeforeText 50pt
        \vskip \SpaceBeforeText\ \ 
      \end{minipage}
    \end{posterindent}
    \ifx\poster@centre\poster@empty\else\vfil\fi
    % If not one column, begin a multicols environment
    \if@postercolumns
      \begin{multicols}{\poster@columns}
    \fi
    \begin{posterindent}                % closed by AtEndDocument
  \else
    \typeout{WARNING: you can only call maketitle once per document}
  \fi
}
\let\maketitle\makeposter

%% do the standard header/footer stuff

\newcommand{\ps@firstpage}{%
  \let\@evenhead=\@empty
  \let\@evenfoot=\@empty
  \let\@oddhead=\@empty
  \renewcommand{\@oddfoot}{%
    \small \vbox to \topskip {\vss
%%      \begin{flushright}
%%        \leavevmode 
%%          \resizebox{!}{\poster@deptimageheight}{%
%%            \includegraphics{\poster@deptimage}}
%%      \end{flushright}%
    \vss}%
    % Now we will add an empty box in order to put TeX stuff beyond
    % all the pstricks stuff, otherwise it gets ignored by dvips -E
    \hskip -1pt\hbox{\white\vrule height0pt depth1.5in width1pt}%
  }%
}

\pagestyle{empty}
\thispagestyle{firstpage}

%% redefine sectioning, captioning, etc. to be in colour and unnumbered

\sectionfont{\SectionColour\Huge\bfseries}
\subsectionfont{\SectionColour\Large\bfseries}
\setcounter{secnumdepth}{-1}


\renewcommand{\caption}[1]{%
  \vbox{\begin{posterindent}\begin{center}%
  {\CaptionColour\em #1}\end{center}\end{posterindent}}
}

\renewcommand{\labelitemi}{\ItemizeColour$\bullet$}
\renewcommand{\labelitemii}{\ItemizeColour\bfseries --}
\renewcommand{\labelitemiii}{\ItemizeColour$\ast$}
\renewcommand{\labelitemiv}{\ItemizeColour$\cdot$}

%% provide a simple mechanism to incorporate EPS figures
%% \epsfigure[width-percentage]{filename}{caption}

\newdimen\poster@figwidth
\newcount\poster@figmult

\newcommand{\epsfigure}[3][75]{%
  \poster@figmult = \number#1
  \ifnum#1<5
     \typeout{WARNING: Figure width must be >= 5 percent. Resetting to 75.}
     \poster@figmult = 75
  \else \ifnum#1>100
     \typeout{WARNING: Figure width must be <= 100 percent. Resetting to 75.}
     \poster@figmult = 75
  \fi\fi
  \poster@figwidth = \linewidth
  \divide\poster@figwidth by 100
  \multiply\poster@figwidth by \poster@figmult
  \begin{center}\vbox{%
     \vskip 25pt plus 5pt minus 5pt
     \epsfig{file=#2,width=\poster@figwidth}
     \vskip 25pt plus 5pt minus 5pt
     \caption{#3}}
  \end{center}%
}

%% Turn off page numbering and setup paragraph spacing etc. 

\parindent=0in
\parskip=30pt

%%
%% EOF: ucbposter.cls
%%

