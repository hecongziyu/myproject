# 生成初中、小学阶段数学公式
# https://latex.91maths.com/eg/xxjhtx.html
import os
from latex_txt_utils import latex_add_space
import os
import re
import numpy as np

math_number_latex = [r'%s',]


# 小学阶段
math_formulas_latex_x = [
    r'S=\frac {{1}}{{2}} \left( a+b \left)  \times h\right. \right.',
    r'S=\frac{{1}}{{2}}b',
    r'S=\sqrt{{p{ \left( {p-a} \right) }{ \left( {p-b} \right) }{ \left( {p-c} \right) }}}',
    r'r=\frac{{1}}{{2}}d',
    r'S=\mathop{{{r}}}\nolimits^{{2}} \pi =\frac{{1}}{{4}}\mathop{{{d}}}\nolimits^{{2}} \pi',
    r'C=d \pi =2r \pi',
    r'S=2rh \pi +2\mathop{{{r}}}\nolimits^{{2}} \pi',
    r'V=Sh=\mathop{{{r}}}\nolimits^{{2}} \pi h',
    r'S= \pi rl+\frac{{1}}{{2}} \alpha l\mathop{{}}\nolimits^{{2}}',
    r'V=\frac{{1}}{{3}}Sh=\frac{{1}}{{3}}\mathop{{{r}}}\nolimits^{{2}} \pi h',
    r'V=a \times a \times a',
    r'C= \left( a+b \left)  \times 2\right. \right. ',
    r'1\frac{{2}}{{3}}',
    r'\frac{{5}}{{6}} \times \frac{{1}}{{3}}=\frac{{5 \times 1}}{{6 \times 3}}=\frac{{5}}{{18}}',
    r'\frac{{2}}{{5}} \times \frac{{1}}{{4}}=\frac{{2 \times 1}}{{5 \times 4}}=\frac{{2}}{{20}}=\frac{{1}}{{10}}',
    r'\frac{{4}}{{5}} \times 3=\frac{{4 \times 3}}{{5}}=\frac{{12}}{{5}}',
    r'\frac{{3}}{{8}} \times 2=\frac{{3 \times 2}}{{8}}=\frac{{6}}{{8}}=\frac{{3}}{{4}}',
    r'\frac{{3}}{{8}} \div 2=\frac{{3}}{{8}} \times \frac{{1}}{{2}}=\frac{{3}}{{16}}',
    r'\frac{{3}}{{10}} \div \frac{{2}}{{5}}=\frac{{3}}{{10}} \times \frac{{5}}{{2}}=\frac{{15}}{{20}}=\frac{{3}}{{4}}',
    r'\frac{{4}}{{5}}-\frac{{1}}{{5}}=\frac{{4-1}}{{5}}=\frac{{3}}{{5}}',
    r'\frac{{1}}{{2}}+\frac{{1}}{{3}}=\frac{{1 \times 3}}{{2 \times 3}}+\frac{{1 \times 2}}{{3 \times 2}}=\frac{{3+2}}{{6}}=\frac{{5}}{{6}}',
    r'\left( a+b \left)  \times c=a \times c+b \times c\right. \right.',
    r'\left( a-b \left)  \times c=a \times c-b \times c\right. \right.',
    r'a \times b=b \times a',
    r'a \times b \times c=a \times  \left( b \times c \left) = \left( a \times c \left)  \times b\right. \right. \right. \right. ',
    r'a \div b= \left( a \times c \left)  \div  \left( b \times c \right) \right. \right.',
    r'a \div b= \left( a \div c \left)  \div  \left( b \div c \right) \right. \right. ',
    r'a+c=b+c',
    r'a-c=b-c',
    r'a \times c=b \times c',
    r'a \div c=b \div c \left( c \neq 0 \right)',
    r'\mathop{{{a}}}\nolimits^{{c}}=\mathop{{{b}}}\nolimits^{{c}}',        
    r'\sqrt[{c}]{{a}}=\sqrt[{c}]{{b}}',
    r'\mathop{{a}}\nolimits_{{1}}=\mathop{{a}}\nolimits_{{2}}=\mathop{{a}}\nolimits_{{3}}=\mathop{{a}}\nolimits_{{4}}= \cdots  \cdots =\mathop{{a}}\nolimits_{{n-1}}=\mathop{{a}}\nolimits_{{n}}',
    r'a+b+c=a+ \left( b+c \left) = \left( a+c \left) +b\right. \right. \right. \right.'
]

# 初中阶段
math_formulas_latex_m = [
    r'\frac{{a}}{{b}} \pm \frac{{c}}{{d}}=\frac{{ad \pm bc}}{{bd}}',
    r'\sqrt[{n}]{{ab}}=\sqrt[{n}]{{a}} \cdot \sqrt[{n}]{{b}},a \ge 0,b \ge 0',
    r'\frac{{a}}{{b}}=\frac{{c}}{{d}} \Rightarrow {\frac{{b}}{{a}}=\frac{{d}}{{c}}}',
    r'\frac{{a}}{{b}}=\frac{{c}}{{d}} \Rightarrow {\frac{{a-b}}{{b}}=\frac{{c-d}}{{d}}}',
    r'\mathop{{ \left( {\frac{{a}}{{b}}} \right) }}\nolimits^{{n}}=\frac{{\mathop{{a}}\nolimits^{{n}}}}{{\mathop{{b}}\nolimits^{{n}}}}',
    r'\sqrt[{n}]{{\frac{{a}}{{b}}}}=\frac{{\sqrt[{n}]{{a}}}}{{\sqrt[{n}]{{b}}}},a \ge 0,b \ge 0',
    r'\sqrt[{n}]{{\frac{{a}}{{b}}}}=\frac{{\sqrt[{n}]{{a}}}}{{\sqrt[{n}]{{b}}}}{ \left( {a > 0,b > 0} \right) }',
    r'\frac{{a}}{{b}} \cdot \frac{{c}}{{d}}=\frac{{ac}}{{bd}}',
    r'\sqrt[{n}]{{\mathop{{a}}\nolimits^{{m}}}}=\mathop{{ \left( {\sqrt[{n}]{{a}}} \right) }}\nolimits^{{m}},a \ge 0',
    r'\sqrt[{n}]{{\mathop{{a}}\nolimits^{{n}}}}={\mathop{{ \left( {\sqrt[{n}]{{a}}} \right) }}\nolimits^{{n}}=a}',
    r'\sqrt[{np}]{{\mathop{{a}}\nolimits^{{mp}}}}=\sqrt[{n}]{{\mathop{{a}}\nolimits^{{m}}}},a \ge 0',
    r'\frac{{1}}{{\sqrt{{a}}}}=\frac{{\sqrt{{a}}}}{{a}},a \ge 0',
    r'\begin{array}{l l}{a > 0,b > 0,a \neq b,c > 0,d > 0} \\ {\frac{{\sqrt{{c}}+\sqrt{{d}}}}{{\sqrt{{a}}-\sqrt{{b}}}}=\frac{{ \left( {\sqrt{{c}}+\sqrt{{d}}} \left) { \left( {\sqrt{{a}}+\sqrt{{b}}} \right) }\right. \right. }}{{a-b}}} \end{array}',
    r'\begin{array}{l l}{a > 0,b > 0,a \neq b,c > 0,d > 0}\\{\frac{{\sqrt{{c}}+\sqrt{{d}}}}{{\sqrt{{a}}+\sqrt{{b}}}}=\frac{{ \left( {\sqrt{{c}}+\sqrt{{d}}} \left) { \left( {\sqrt{{a}}-\sqrt{{b}}} \right) }\right. \right. }}{{a-b}}}\end{array}',
    r'\begin{array}{l l}{a \neq 0,b \neq 0,c \neq 0,d \neq 0}\\{\frac{{a}}{{b}}=\frac{{c}}{{d}} \Rightarrow {\frac{{a}}{{c}}=\frac{{b}}{{d}}}}\end{array}',
    r'\begin{array}{l l}{b \neq 0,d \neq 0}\\{\frac{{a}}{{b}}=\frac{{c}}{{d}} \Rightarrow {\frac{{a+b}}{{b}}=\frac{{c+d}}{{d}}}}\end{array}',
    r'\frac{{a}}{{b}} \pm \frac{{c}}{{b}}=\frac{{a \pm c}}{{b}}',
    r'\begin{array}{l l}{b \neq 0,d \neq 0}\\{\frac{{a}}{{b}}=\frac{{c}}{{d}} \Rightarrow ad=bc}\end{array}',
    r'y\mathop{{}}\nolimits^{{2}}=2px\text{,} \left( p > 0 \right) ',
    r'y\mathop{{}}\nolimits^{{2}}=-2px\text{,} \left( p > 0 \right) ',
    r'x\mathop{{}}\nolimits^{{2}}=2py\text{,} \left( p > 0 \right) ',
    r'x\mathop{{}}\nolimits^{{2}}=-2py\text{,} \left( p > 0 \right)',
    r'\left( x-a \left) \mathop{{}}\nolimits^{{2}}+ \left( y-b \left) \mathop{{}}\nolimits^{{2}}=r\mathop{{}}\nolimits^{{2}}\right. \right. \right. \right.',
    r'ax+ay+bx+by \\= \left ( a+b \right )x+\left ( a+b \right )y \\=\left ( a+b \right )\left ( x+y \right )',
    r'\mathop{{a}}\nolimits^{{3}}-\mathop{{b}}\nolimits^{{3}}={ \left( {a-b} \right) }{ \left( {\mathop{{a}}\nolimits^{{2}}+ab+\mathop{{b}}\nolimits^{{2}}} \right) }',
    r'\mathop{{a}}\nolimits^{{3}}+\mathop{{b}}\nolimits^{{3}}={ \left( {a+b} \right) }{ \left( {\mathop{{a}}\nolimits^{{2}}-ab+\mathop{{b}}\nolimits^{{2}}} \right) }',
    r'ax\mathop{{}}\nolimits^{{2}}+bx+c=a \left[ x- \left( \frac{{-b+\sqrt{{b\mathop{{}}\nolimits^{{2}}-4ac}}}}{{2a}} \left)  \left]  \left[ x- \left( \frac{{-b-\sqrt{{b\mathop{{}}\nolimits^{{2}}-4ac}}}}{{2a}} \left)  \right] \right. \right. \right. \right. \right. \right. ',
    r'\mathop{{a}}\nolimits^{{2}}-\mathop{{b}}\nolimits^{{2}}={ \left( {a+b} \right) }{ \left( {a-b} \right) }',
    r'\mathop{{ \left( {a+b+c} \right) }}\nolimits^{{2}}=\mathop{{a}}\nolimits^{{2}}+\mathop{{b}}\nolimits^{{2}}+\mathop{{c}}\nolimits^{{2}}+2ab+2bc+2ac',
    r' \left( {x+a} \left) { \left( {x+b} \right) }=\mathop{{x}}\nolimits^{{2}}+{ \left( {a+b} \right) }x+ab\right. \right. ',
    r'\begin{array}{l l}{\mathop{{ \left( {a+b} \right) }}\nolimits^{{3}}=\mathop{{a}}\nolimits^{{3}}+3\mathop{{a}}\nolimits^{{2}}b+3a\mathop{{b}}\nolimits^{{2}}+\mathop{{b}}\nolimits^{{3}}}\\{\mathop{{ \left( {a-b} \right) }}\nolimits^{{3}}=\mathop{{a}}\nolimits^{{3}}-3\mathop{{a}}\nolimits^{{2}}b+3a\mathop{{b}}\nolimits^{{2}}-\mathop{{b}}\nolimits^{{3}}}\end{array}',
    r'\begin{array}{l l}{\mathop{{ \left( {a+b} \right) }}\nolimits^{{2}}=\mathop{{a}}\nolimits^{{2}}+2ab+\mathop{{b}}\nolimits^{{2}}}\\{\mathop{{ \left( {a-b} \right) }}\nolimits^{{2}}=\mathop{{a}}\nolimits^{{2}}-2ab+\mathop{{b}}\nolimits^{{2}}}\end{array}',
    r'\mathop{{a}}\nolimits^{{2}}+\mathop{{b}}\nolimits^{{2}}=\mathop{{c}}\nolimits^{{2}}',
    r'l=\frac{{n \pi r}}{{180}}',
    r'd=\sqrt{{\mathop{{ \left( {\mathop{{x}}\nolimits_{{2}}-\mathop{{x}}\nolimits_{{1}}} \right) }}\nolimits^{{2}}+\mathop{{ \left( {\mathop{{y}}\nolimits_{{2}}-\mathop{{y}}\nolimits_{{1}}} \right) }}\nolimits^{{2}}}}',
    r'S=4 \pi r\mathop{{}}\nolimits^{{2}}',
    r'V=\frac{{4}}{{3}} \pi r\mathop{{}}\nolimits^{{3}}',
    r'S=\frac{{lr}}{{2}}',
    r'S=\frac{{1}}{{2}}ab{ \text{sin}  \theta }',
    r'V=\frac{{1}}{{3}} \pi { \left( {r\mathop{{}}\nolimits_{{1}}\mathop{{}}\nolimits^{{2}}+r\mathop{{}}\nolimits_{{1}}r\mathop{{}}\nolimits_{{2}}+r\mathop{{}}\nolimits_{{2}}\mathop{{}}\nolimits^{{2}}} \left) h\right. \right. }',
    r'S=\frac{{1}}{{2}} \left( C\mathop{{}}\nolimits_{{1}}+C\mathop{{}}\nolimits_{{2}} \left) h\right. \right.',
    r'\mathop{{ \text{sin} }}\nolimits^{{2}}\frac{{ \alpha }}{{2}}=\frac{{1- \text{cos}  \alpha }}{{2}}',
    r'\mathop{{ \text{cos} }}\nolimits^{{2}}\frac{{ \alpha }}{{2}}=\frac{{1+ \text{cos}  \alpha }}{{2}}',
    r'\text{tan} \frac{{ \alpha }}{{2}}=\frac{{ \text{sin}  \alpha }}{{1+ \text{cos}  \alpha }}',
    r'\text{tan} 2 \alpha =\frac{{2 \text{tan}  \alpha }}{{1-\mathop{{ \text{tan} }}\nolimits^{{2}} \alpha }}r',
    r'\text{sin}  \alpha + \text{sin}  \beta =2 \text{sin} \frac{{ \alpha + \beta }}{{2}} \text{cos} \frac{{ \alpha - \beta }}{{2}}',
    r'\text{sin}  \alpha - \text{sin}  \beta =2 \text{cos} \frac{{ \alpha + \beta }}{{2}} \text{sin} \frac{{ \alpha - \beta }}{{2}}',
    r'\text{cos}  \alpha + \text{cos}  \beta =2 \text{cos} \frac{{ \alpha + \beta }}{{2}} \text{cos} \frac{{ \alpha - \beta }}{{2}}',
    r'\text{cos}  \alpha - \text{cos}  \beta =-2 \text{sin} \frac{{ \alpha + \beta }}{{2}} \text{sin} \frac{{ \alpha - \beta }}{{2}}',
    r'2 \text{cos}  \alpha  \text{cos}  \beta = \text{cos} { \left( { \alpha - \beta } \right) }+ \text{cos} { \left( { \alpha + \beta } \right) }',
    r'2 \text{sin}  \alpha  \text{sin}  \alpha  \beta = \text{cos} { \left( { \alpha + \beta } \right) }- \text{cos} { \left( { \alpha + \beta } \right) }',
    r'2 \text{sin}  \alpha  \text{cos}  \beta = \text{sin} { \left( { \alpha - \beta } \right) }+ \text{sin} { \left( { \alpha + \beta } \right) }',
    r'\text{sin} { \left( { \alpha + \beta } \right) }= \text{sin}  \alpha  \text{cos}  \beta + \text{cos}  \alpha  \text{sin}  \beta',
    r'\text{sin} { \left( { \alpha - \beta } \right) }= \text{sin}  \alpha  \text{cos}  \beta - \text{cos}  \alpha  \text{sin}  \beta',
    r'\text{cos} { \left( { \alpha + \beta } \right) }= \text{cos}  \alpha  \text{cos}  \beta - \text{sin}  \alpha  \text{sin}  \beta',
    r'\text{cos} { \left( { \alpha - \beta } \right) }= \text{cos}  \alpha  \text{cos}  \beta + \text{sin}  \alpha  \text{sin}  \beta',
    r'\text{tan} { \left( { \alpha + \beta } \right) }=\frac{{ \text{tan}  \alpha + \text{tan}  \beta }}{{1- \text{tan}  \alpha  \text{tan}  \beta }}',
    r'\text{tan} { \left( { \alpha - \beta } \right) }=\frac{{ \text{tan}  \alpha - \text{tan}  \beta }}{{1- \text{tan}  \alpha  \text{tan}  \beta }}',
    r'360 \circ =2 \pi rad',
    r'180 \circ = \pi rad',
    r'1 \circ =\frac{{ \pi }}{{180}}rad \approx 0.1745rad',
    r'1rad=\frac{{180 \circ }}{{ \pi }} \approx 57.30 \circ',
    r'\mathop{{ \text{sin} }}\nolimits^{{2}} \alpha +\mathop{{ \text{cos} }}\nolimits^{{2}}=1',
    r'\mathop{{ \text{tan} }}\nolimits^{{2}} \alpha +1=\mathop{{ \text{sec} }}\nolimits^{{2}} \alpha',
    r'\mathop{{ \text{cot} }}\nolimits^{{2}} \alpha +1=\mathop{{ \text{csc} }}\nolimits^{{2}} \alpha',
    r'\text{sin} { \left( {2k \pi + \alpha } \left) = \text{sin}  \alpha k \in  \mathbb{Z} \right. \right. }',
    r'\text{cos} { \left( {2k \pi + \alpha } \left) = \text{cos}  \alpha k \in  \mathbb{Z} \right. \right. }',
    r'\text{tan} { \left( {2k \pi + \alpha } \left) = \text{tan}  \alpha k \in  \mathbb{Z} \right. \right. }',
    r'\text{cot} { \left( {2k \pi + \alpha } \left) = \text{cot}  \alpha k \in  \mathbb{Z} \right. \right. }',
    r'\text{sec} { \left( {2k \pi + \alpha } \left) = \text{sec}  \alpha k \in  \mathbb{Z} \right. \right. }',
    r'\text{csc} { \left( {2k \pi + \alpha } \left) = \text{csc}  \alpha k \in  \mathbb{Z} \right. \right. }',
    r'\text{sin} { \left( { \pi + \alpha } \left) =- \text{sin}  \alpha \right. \right. }',
    r'\text{cos} { \left( { \pi + \alpha } \left) =- \text{cos}  \alpha \right. \right. }',
    r'\text{tan} { \left( { \pi + \alpha } \left) = \text{tan}  \alpha \right. \right. }',
    r'\text{cot} { \left( { \pi + \alpha } \left) = \text{cot}  \alpha \right. \right. }',
    r'\text{sec} { \left( { \pi + \alpha } \left) =- \text{sec}  \alpha \right. \right. }',
    r'\text{csc} { \left( { \pi + \alpha } \left) =- \text{csc}  \alpha \right. \right. }',
    r'\frac{{ \text{sin} A}}{{a}}=\frac{{ \text{sin} B}}{{b}}=\frac{{ \text{sin} C}}{{c}}=\frac{{1}}{{2R}}',
    r'\mathop{{a}}\nolimits^{{2}}=\mathop{{b}}\nolimits^{{2}}+\mathop{{c}}\nolimits^{{2}}-2bc{ \text{cos} A}',
    r'\text{sin} { \left( {\frac{{ \pi }}{{2}}- \alpha } \left) = \text{cos}  \alpha \right. \right. }',
    r'\text{cos} { \left( {\frac{{ \pi }}{{2}}- \alpha } \left) = \text{sin}  \alpha \right. \right. }',
    r'\text{tan} { \left( {\frac{{ \pi }}{{2}}- \alpha } \left) = \text{cot}  \alpha \right. \right. }',
    r'{ \left| a+b \left|  \le  \left| a \left| + \left| b \right| \right. \right. \right. \right. }',
    r'\left| a-b \left|  \le  \left| a \left| + \left| b \right| \right. \right. \right. \right. ',
    r'\left| a \left|  \le b \Leftrightarrow -b \le a \le b\right. \right.',
    r' \left| a-b \left|  \ge  \left| a \left| - \left| b \right| \right. \right. \right. \right. ',
    r'- \left| a \left|  \le a \le  \left| a \right| \right. \right. ',
    r's=\sqrt{{s\mathop{{}}\nolimits^{{2}}}}=\sqrt{{\frac{{1}}{{n}} \left[  \left( x\mathop{{}}\nolimits_{{1}}- \overline {x} \left) \mathop{{}}\nolimits^{{2}}+ \left( x\mathop{{}}\nolimits_{{2}}- \overline {x} \left) \mathop{{}}\nolimits^{{2}}+ \cdots + \left( x\mathop{{}}\nolimits_{{n}}- \overline {x} \left) \mathop{{}}\nolimits^{{2}} \right] \right. \right. \right. \right. \right. \right. }}',
    r's\mathop{{}}\nolimits^{{2}}=\frac{{1}}{{n}} \left[  \left( x\mathop{{}}\nolimits_{{1}}- \overline {x} \left) \mathop{{}}\nolimits^{{2}}+ \left( x\mathop{{}}\nolimits_{{2}}- \overline {x} \left) \mathop{{}}\nolimits^{{2}}+ \cdots + \left( x\mathop{{}}\nolimits_{{n}}- \overline {x} \left) \mathop{{}}\nolimits^{{2}} \right] \right. \right. \right. \right. \right. \right.',
    r'\overline {x}=\frac{{x\mathop{{}}\nolimits_{{1}}w\mathop{{}}\nolimits_{{1}}+x\mathop{{}}\nolimits_{{2}}w\mathop{{}}\nolimits_{{2}}+ \cdots +x\mathop{{}}\nolimits_{{n}}w\mathop{{}}\nolimits_{{n}}}}{{w\mathop{{}}\nolimits_{{1}}+w\mathop{{}}\nolimits_{{2}}+ \cdots +w\mathop{{}}\nolimits_{{n}}}}',
    r' \overline {x}=\frac{{1}}{{n}} \left( x\mathop{{}}\nolimits_{{1}}+x\mathop{{}}\nolimits_{{2}}+ \cdots +x\mathop{{}}\nolimits_{{n}} \right)',
    r'1+2+3+4+5+ \cdots +n=\frac{{n \left( n+1 \right) }}{{2}}',
    r'1+3+5+7+9+ \cdots + \left( 2n-1 \left) =n\mathop{{}}\nolimits^{{2}}\right. \right. ',
    r'2+4+6+8+10+ \cdots +2n=n \left( n+1 \right) ',
    r'1\mathop{{}}\nolimits^{{2}}+2\mathop{{}}\nolimits^{{2}}+3\mathop{{}}\nolimits^{{2}}+4\mathop{{}}\nolimits^{{2}}+5\mathop{{}}\nolimits^{{2}}+ \cdots +n\mathop{{}}\nolimits^{{2}}=\frac{{n \left( n+1 \left)  \left( 2n+1 \right) \right. \right. }}{{6}}',
    r'1\mathop{{}}\nolimits^{{3}}+2\mathop{{}}\nolimits^{{3}}+3\mathop{{}}\nolimits^{{3}}+4\mathop{{}}\nolimits^{{3}}+5\mathop{{}}\nolimits^{{3}}+ \cdots +n\mathop{{}}\nolimits^{{3}}=\frac{{n\mathop{{}}\nolimits^{{2}} \left( n+1 \left) \mathop{{}}\nolimits^{{2}}\right. \right. }}{{4}}',
    r'1 \times 2+2 \times 3+3 \times 4+4 \times 5+ \cdots +n \left( n+1 \left) =\frac{{n \left( n+1 \left)  \left( n+2 \right) \right. \right. }}{{3}}\right. \right. ',
    r'\mathop{{x}}\nolimits_{{1}}=\frac{{-b+\sqrt{{\mathop{{b}}\nolimits^{{2}}-4ac}}}}{{2a}}',
    r'\mathop{{x}}\nolimits_{{2}}=\frac{{-b-\sqrt{{\mathop{{b}}\nolimits^{{2}}-4ac}}}}{{2a}}']

# 高中阶段
math_formulas_latex_g = [
    r'P \left( X=k \left) =C\mathop{{}}\nolimits_{{n}}^{{k}}p\mathop{{}}\nolimits^{{k}} \left( 1-p \left) \mathop{{}}\nolimits^{{n-k}}\text{（}k=1,2, \cdots ,n\text{）}\right. \right. \right. \right. ',
    r'P \left( AB \left) =P \left( A \left) P \left( B \right) \right. \right. \right. \right. ',
    r'\begin{array}{l l}{\begin{array}{l l}{X \sim B \left( n,p \right) }\\{P{ \left( {X=k} \right) }=C\mathop{{}}\nolimits_{{n}}^{{k}}p\mathop{{}}\nolimits^{{k}} \left( 1-p \left) \mathop{{}}\nolimits^{{n-k}}\text{（}k=0,1,2, \cdots ,n\text{）}\right. \right. }\end{array}}\\{EX=np}\\{DX=np \left( 1-p \right) }\end{array}',
    r'P \left( A \left) +P \left(  \overline {A} \left) =1\right. \right. \right. \right.',
    r'A \cup B= \left\{ x \left| x \in A,\text{或}x \in B \right\} \right. ',
    r'C\mathop{{}}\nolimits_{{U}}A= \left\{ x \left| x \in U,\text{且}x \notin A \right\} \right. ',
    r'A \subseteq B,B \subseteq A \Leftrightarrow A=B',
    r'A \cap B= \left\{ x \left| x \in A,\text{且}x \in B \right\} \right. ',
    r'\mathbb{N} ={ \left\{ {0,1,2, \cdots ,n, \cdots } \right\} }',
    r'\mathbb{Z} = \left\{ { \cdots ,-n, \cdots ,-2,-1,0,1,2, \cdots ,n, \cdots } \right\}',
    r'\mathbb{Q} = \left\{ {\frac{{p}}{{q}} \left| p \in  \mathbb{Z} ,q \in \mathop{{ \mathbb{N} }}\nolimits^{{+}}\text{a}\text{n}\text{d}p \perp q\right. } \right\}',
    r'\mathbb{R}',
    r'\mathop{{ \mathbb{R} }}\nolimits^{{ \ast }}',
    r'\mathop{{ \mathbb{R} }}\nolimits^{{+}}',
    r'\mathop{{ \mathbb{R} }}\nolimits^{{2}}= \mathbb{R}  \times  \mathbb{R} ={ \left\{ { \left( {x,y} \left)  \left| x \in  \mathbb{R} ,y \in  \mathbb{R} \right. \right. \right. } \right\} }',
    r'p\mathop{{}}\nolimits_{{i}} \ge 0\text{（}i=1,2, \cdots ,n\text{）}',
    r'p\mathop{{}}\nolimits_{{1}}+p\mathop{{}}\nolimits_{{2}}+ \cdots +p\mathop{{}}\nolimits_{{n}}=1',
    r'P{ \left( {B \left| A\right. } \right) }=\frac{{P \left( AB \right) }}{{P \left( A \right) }}',
    r'P{ \left( {a < X \le b} \right) }=\mathop{ \int }\nolimits_{{a}}^{{b}} \varphi { \left( {x} \right) }dx',
    r'a \parallel c,b \parallel c \Rightarrow a \parallel b',
    r'l \perp  \beta ,l \subset  \alpha  \Rightarrow  \alpha  \perp  \beta ',
    r'\alpha  \perp  \beta , \alpha  \cap  \beta =l,a \subset  \alpha ,a \perp l \Rightarrow a \perp  \beta ',
    r'\begin{array}{l l}{a \subset  \beta ,b \subset  \beta ,a \cap b=P}\\{a \parallel  \partial ,b \parallel  \partial }\end{array} \left\}  \Rightarrow  \beta  \parallel  \alpha \right. ',
    r'\alpha  \parallel  \beta , \gamma  \cap  \alpha =a, \gamma  \cap  \beta =b \Rightarrow a \parallel b',
    r'A \in l,B \in l,A \in  \alpha ,B \in  \alpha  \Rightarrow l \subset  \alpha',
    r'P \in  \alpha ,P \in  \beta , \alpha  \cap  \beta =l \Rightarrow P \in l',
    r'\begin{array}{l l}{m \subset  \alpha ,n \subset  \alpha ,m \cap n=P}\\{a \perp m,a \perp n}\end{array} \left\} a \perp  \alpha \right. ',
    r'\begin{array}{l l}{a \perp  \alpha }\\{b \perp  \alpha }\end{array} \left\} a \parallel b\right. ',
    r'a \not\subset  \alpha ,b \subset  \alpha ,a \parallel b \Rightarrow a \parallel  \alpha',
    r'a \parallel  \alpha \text{，}a \subset  \beta , \alpha  \cap  \beta =b \Rightarrow a \parallel b',
    r'\mathop{ \sum }\limits_{{n=0}}^{{ \infty }}a\mathop{{q}}\nolimits^{{n}}=a+aq+a\mathop{{q}}\nolimits^{{2}}+ \cdots +a\mathop{{q}}\nolimits^{{n}}+ \cdots',
    r'a\mathop{{}}\nolimits_{{m}} \cdot a\mathop{{}}\nolimits_{{n}}=a\mathop{{}}\nolimits_{{p}} \cdot a\mathop{{}}\nolimits_{{q}}\text{（}m+n=p+q\text{）}',
    r'\begin{array}{l l}{q=1}&{S\mathop{{}}\nolimits_{{n}}=na\mathop{{}}\nolimits_{{1}}}\\{q \neq 1}&{S\mathop{{}}\nolimits_{{n}}=\frac{{a\mathop{{}}\nolimits_{{1}} \left( 1-q\mathop{{}}\nolimits^{{n}} \right) }}{{1-q}}}\end{array}',
    r'a\mathop{{}}\nolimits_{{n}}=a\mathop{{}}\nolimits_{{1}}q\mathop{{}}\nolimits^{{n-1}}',
    r'a\mathop{{}}\nolimits_{{m}}+a\mathop{{}}\nolimits_{{n}}=a\mathop{{}}\nolimits_{{p}}+a\mathop{{}}\nolimits_{{q}}\text{（}m+n=p+q\text{）}',
    r'a\mathop{{}}\nolimits_{{p}}=q,a\mathop{{}}\nolimits_{{q}}=p,\text{则}a\mathop{{}}\nolimits_{{p+q}}=0',
    r'S\mathop{{}}\nolimits_{{n}}=\frac{{n \left( a\mathop{{}}\nolimits_{{1}}+a\mathop{{}}\nolimits_{{n}} \right) }}{{2}}',
    r'S\mathop{{}}\nolimits_{{n}}=na\mathop{{}}\nolimits_{{1}}+\frac{{n \left( n-1 \right) }}{{2}}d',
    r'a\mathop{{}}\nolimits_{{n}}=a\mathop{{}}\nolimits_{{1}}+ \left( n-1 \left) d\right. \right. ',
    r'\left( {2n+1} \left) !!=\frac{{ \left( {2n+1} \left) !\right. \right. }}{{\mathop{{2}}\nolimits^{{n}}n!}}=1 \cdot 3 \cdot 5 \cdots { \left( {2n+1} \right) },{ \left( {-1} \left) !!=0\right. \right. }\right. \right.',
    r'n!=1 \cdot 2 \cdot 3 \cdots n,0!=1',
    r'\left( {2n} \left) !!=\mathop{{2}}\nolimits^{{n}}n!=2 \cdot 4 \cdot 6 \cdots { \left( {2n} \left) ,0!!=0\right. \right. }\right. \right.',
    r'\mathop{{P}}\nolimits_{{n}}=n!',
    r'\mathop{{A}}\nolimits_{{n}}^{{k}}=\frac{{n!}}{{ \left( {n-k} \left) !\right. \right. }}',
    r'\frac{{1}}{{n \left( n+k \right) }}=\frac{{1}}{{k}}{ \left( {\frac{{1}}{{n}}-\frac{{1}}{{n+k}}} \right) }',
    r'\frac{{1}}{{n\mathop{{}}\nolimits^{{2}}-1}}=\frac{{1}}{{2}}{ \left( {\frac{{1}}{{n-1}}-\frac{{1}}{{n+1}}} \right) }',
    r'\frac{{1}}{{4n\mathop{{}}\nolimits^{{2}}-1}}=\frac{{1}}{{2}}{ \left( {\frac{{1}}{{2n-1}}-\frac{{1}}{{2n+1}}} \right) }',
    r'\frac{{n+1}}{{n \left( n-1 \left)  \cdot 2\mathop{{}}\nolimits^{{n}}\right. \right. }}=\frac{{1}}{{ \left( n-1 \left)  \cdot 2\mathop{{}}\nolimits^{{n-1}}\right. \right. }}-\frac{{1}}{{n \cdot 2\mathop{{}}\nolimits^{{n}}}}',
    r'\frac{{1}}{{\sqrt{{n}}+\sqrt{{n+k}}}}=\frac{{1}}{{k}}{ \left( {\sqrt{{n+k}}-\sqrt{{n}}} \right) }',
    r'n \cdot n!= \left( n+1 \left) !-n!\right. \right. ',
    r'S\mathop{{}}\nolimits_{{n}}=S\mathop{{}}\nolimits_{{n-1}}+a\mathop{{}}\nolimits_{{n}}\text{（}n \ge 2\text{）}',
    r'S\mathop{{}}\nolimits_{{n}}=a\mathop{{}}\nolimits_{{1}}+a\mathop{{}}\nolimits_{{2}}+ \cdots +a\mathop{{}}\nolimits_{{n}}',
    r'a\mathop{{}}\nolimits_{{n}}=S\mathop{{}}\nolimits_{{n}}-S\mathop{{}}\nolimits_{{n-1}}\text{（}n \ge 2\text{）}',
    r'\mathop{{C}}\nolimits_{{n}}^{{k}}={ \left( {\begin{array}{c}{n}\\{k}\end{array}} \left) =\frac{{\mathop{{A}}\nolimits_{{n}}^{{k}}}}{{k!}}=\frac{{n!}}{{ \left( {n-k} \left) ! \cdot k!\right. \right. }}\right. \right. }',
    r'\mathop{{C}}\nolimits_{{n}}^{{0}}=1',
    r'\mathop{{C}}\nolimits_{{n}}^{{k}}={\frac{{n}}{{n-k}}\mathop{{C}}\nolimits_{{n-1}}^{{k}}}',
    r'\mathop{{C}}\nolimits_{{n}}^{{k}}=\mathop{{C}}\nolimits_{{n}}^{{n-k}}',
    r'\mathop{{C}}\nolimits_{{n+1}}^{{k}}=\mathop{{C}}\nolimits_{{n}}^{{n}}+\mathop{{C}}\nolimits_{{n}}^{{k-1}}',
    r'\mathop{{C}}\nolimits_{{n+1}}^{{k}}=\mathop{ \sum }\limits_{{j=0}}^{{k}}\mathop{{C}}\nolimits_{{n-j}}^{{k-j}}',
    r'\mathop{{C}}\nolimits_{{n+k+1}}^{{n+1}}=\mathop{ \sum }\limits_{{j=0}}^{{k}}\mathop{{C}}\nolimits_{{n+j}}^{{n}}',
    r'\mathop{{C}}\nolimits_{{m+n}}^{{k}}=\mathop{ \sum }\limits_{{j=0}}^{{k}}\mathop{{C}}\nolimits_{{m}}^{{j}}\mathop{{C}}\nolimits_{{n}}^{{k-j}}',
    r'd={ \left| { \overrightarrow {MN}} \right| }\text{c}\text{o}\text{s}{ \left\langle { \overrightarrow {MN},}\mathop{{n}}\limits^{{ \to }} \right\rangle }=\frac{{ \left| { \overrightarrow {MN} \cdot \mathop{{n}}\limits^{{ \to }}} \right| }}{{ \left| {\mathop{{n}}\limits^{{ \to }}} \right| }}',
    r'\text{sin} \theta = \left| \text{c}\text{o}\text{s}{ \left\langle {\mathop{{a}}\limits^{{ \to }},\mathop{{n}}\limits^{{ \to }}} \right\rangle } \right|',
    r'text{cos} \theta = \left| \text{c}\text{o}\text{s}{ \left\langle {\mathop{{a}}\limits^{{ \to }},\mathop{{b}}\limits^{{ \to }}} \right\rangle } \right|',
    r'\mathop{{a}}\limits^{{ \to }}={ \left( {x\mathop{{}}\nolimits_{{1}},y\mathop{{}}\nolimits_{{1}}} \right) },\mathop{{b}}\limits^{{ \to }}={ \left( {x\mathop{{}}\nolimits_{{2}},y\mathop{{}}\nolimits_{{2}}} \right) }',
    r'\lambda  \left( \mathop{{a}}\limits^{{ \to }}+\mathop{{b}}\limits^{{ \to }} \left) = \lambda \mathop{{a}}\limits^{{ \to }}+ \lambda \mathop{{b}}\limits^{{ \to }}\right. \right. ',
    r'\left(  \lambda + \mu  \left) \mathop{{a}}\limits^{{ \to }}= \lambda \mathop{{a}}\limits^{{ \to }}+ \mu \mathop{{a}}\limits^{{ \to }}\right. \right. ',
    r'\lambda  \left(  \mu \mathop{{a}}\limits^{{ \to }} \left) = \left(  \lambda  \mu  \left) \mathop{{a}}\limits^{{ \to }}\right. \right. \right. \right. ',
    r'\mathop{{a}}\limits^{{ \to }}={ \left( {x\mathop{{}}\nolimits_{{1}},y\mathop{{}}\nolimits_{{1}}} \right) },\mathop{{b}}\limits^{{ \to }}={ \left( {x\mathop{{}}\nolimits_{{2}},y\mathop{{}}\nolimits_{{2}}} \right) },\mathop{{a}}\limits^{{ \to }} \neq \mathop{{0}}\limits^{{ \to }},\mathop{{b}}\limits^{{ \to }} \neq \mathop{{0}}\limits^{{ \to }}',
    r'\mathop{{a}}\limits^{{ \to }} \perp \mathop{{b}}\limits^{{ \to }} \Leftrightarrow \mathop{{a}}\limits^{{ \to }} \cdot \mathop{{b}}\limits^{{ \to }}=0 \Leftrightarrow x\mathop{{}}\nolimits_{{1}}x\mathop{{}}\nolimits_{{2}}+y\mathop{{}}\nolimits_{{1}}y\mathop{{}}\nolimits_{{2}}=0',
    r'\left| \mathop{{a}}\limits^{{ \to }} \left| =\sqrt{{x\mathop{{}}\nolimits^{{2}}+y\mathop{{}}\nolimits^{{2}}}}\right. \right. ',
    r'\mathop{{a}}\limits^{{ \to }} \cdot \mathop{{b}}\limits^{{ \to }}= \left| \mathop{{a}}\limits^{{ \to }} \left|  \left| \mathop{{b}}\limits^{{ \to }} \left| \text{c}\text{o}\text{s} \theta \right. \right. \right. \right. ',
    r'\mathop{{a}}\limits^{{ \to }}+\mathop{{b}}\limits^{{ \to }}= \left( x\mathop{{}}\nolimits_{{1}}+x\mathop{{}}\nolimits_{{2}},y\mathop{{}}\nolimits_{{1}}+y\mathop{{}}\nolimits_{{2}} \right)',
    r'\mathop{{a}}\limits^{{ \to }}-\mathop{{b}}\limits^{{ \to }}= \left( x\mathop{{}}\nolimits_{{1}}-x\mathop{{}}\nolimits_{{2}},y\mathop{{}}\nolimits_{{1}}-y\mathop{{}}\nolimits_{{2}} \right)',
    r'\mathop{{a}}\limits^{{ \to }} \cdot \mathop{{b}}\limits^{{ \to }}=x\mathop{{}}\nolimits_{{1}}x\mathop{{}}\nolimits_{{2}}+y\mathop{{}}\nolimits_{{1}}y\mathop{{}}\nolimits_{{2}}',
    r'\lambda \mathop{{a}}\limits^{{ \to }}= \left(  \lambda x, \lambda y \right)',
    r'\mathop{{a}}\limits^{{ \to }} \cdot \mathop{{a}}\limits^{{ \to }}= \left| \mathop{{a}}\limits^{{ \to }} \left| \mathop{{}}\nolimits^{{2}}\right. \right. ',
    r'\mathop{{a}}\limits^{{ \to }}={ \left( {x\mathop{{}}\nolimits_{{1}},y\mathop{{}}\nolimits_{{1}}} \right) },\mathop{{b}}\limits^{{ \to }}={ \left( {x\mathop{{}}\nolimits_{{2}},y\mathop{{}}\nolimits_{{2}}} \right) }\text{，}\mathop{{b}}\limits^{{ \to }} \neq \mathop{{0}}\limits^{{ \to }}',
    r'\frac{{x\mathop{{}}\nolimits^{{2}}}}{{a\mathop{{}}\nolimits^{{2}}}}-\frac{{y\mathop{{}}\nolimits^{{2}}}}{{b\mathop{{}}\nolimits^{{2}}}}=1',
    r'\frac{{x\mathop{{}}\nolimits^{{2}}}}{{a\mathop{{}}\nolimits^{{2}}}}+\frac{{y\mathop{{}}\nolimits^{{2}}}}{{b\mathop{{}}\nolimits^{{2}}}}=1',
    r'\left\{\begin{array}{l l}{x=a\text{c}\text{o}\text{s} \theta }\\{y=b\text{s}\text{i}\text{n} \theta }\end{array}\right. ',
    r'\left\{ \begin{array}{l l}{x=a+r\text{c}\text{o}\text{s} \theta }\\{y=b+r\text{s}\text{i}\text{n} \theta }\end{array}\right. ',
    r'y-y\mathop{{}}\nolimits_{{1}}=k \left( x-x\mathop{{}}\nolimits_{{1}} \right) ',
    r'\frac{{x}}{{a}}+\frac{{y}}{{b}}=1\text{()}a \neq 0,b \neq 0\text{)}',
    r'\frac{{y-y\mathop{{}}\nolimits_{{1}}}}{{y\mathop{{}}\nolimits_{{2}}-y\mathop{{}}\nolimits_{{1}}}}=\frac{{x-x\mathop{{}}\nolimits_{{1}}}}{{x\mathop{{}}\nolimits_{{2}}-x\mathop{{}}\nolimits_{{1}}}}\text{()}x\mathop{{}}\nolimits_{{1}} \neq x\mathop{{}}\nolimits_{{2}},y\mathop{{}}\nolimits_{{1}} \neq y\mathop{{}}\nolimits_{{2}}\text{)}',
    r' \left( x\mathop{{}}\nolimits_{{2}}-x\mathop{{}}\nolimits_{{1}} \left)  \left( y-y\mathop{{}}\nolimits_{{1}} \left) - \left( y\mathop{{}}\nolimits_{{2}}-y\mathop{{}}\nolimits_{{1}} \left)  \left( x-x\mathop{{}}\nolimits_{{1}} \left) =0\right. \right. \right. \right. \right. \right. \right. \right. ',
    r'y=kx+b',
    r'{f \prime { \left( {\mathop{{x}}\nolimits_{{0}}} \right) }=\mathop{{ \text{lim} }}\limits_{{ \Delta x \to 0}}\frac{{ \Delta y}}{{ \Delta x}}=\mathop{{ \text{lim} }}\limits_{{ \Delta x \to 0}}\frac{{f{ \left( {\mathop{{x}}\nolimits_{{0}}+ \Delta x} \right) }-f{ \left( {\mathop{{x}}\nolimits_{{0}}} \right) }}}{{ \Delta x}}}',
    r'\mathop{{\left. y \prime  \right| }}\nolimits_{{x=\mathop{{x}}\nolimits_{{0}}}}',
    r'\mathop{{\left.  \dot {y} \right| }}\nolimits_{{x=\mathop{{x}}\nolimits_{{0}}}}',
    r' \left[ {\mathop{{f}}\nolimits^{{-1}}{ \left( {x} \right) }} \left]  \prime =\frac{{1}}{{f \prime { \left( {y} \right) }}}\right. \right. ',
    r'y=f{ \left( {u} \right) },u=g{ \left( {x} \right) }',
    r'\frac{{ \text{d} y}}{{ \text{d} x}}=\frac{{ \text{d} y}}{{ \text{d} u}} \cdot \frac{{ \text{d} u}}{{ \text{d} x}}',
    r'\left( {\mathop{{x}}\nolimits^{{ \mu }}} \left)  \prime = \mu \mathop{{x}}\nolimits^{{ \mu -1}}\right. \right.',
    r'\left( {C} \left)  \prime =0\right. \right.',
    r'\left( { \text{sin} x} \left)  \prime = \text{cos} x\right. \right.',
    r'\left( { \text{cos} x} \left)  \prime =- \text{sin} x\right. \right.',
    r'\left( { \text{tan} x} \left)  \prime =\mathop{{ \text{sec} }}\nolimits^{{2}}x\right. \right.',
    r'\left( { \text{cot} x} \left)  \prime =-\mathop{{ \text{csc} }}\nolimits^{{2}}x\right. \right.',
    r'\left( { \text{sec} x} \left)  \prime = \text{sec} x \text{tan} x\right. \right.',
    r'\left( { \text{csc} x} \left)  \prime =- \text{csc} x{ \text{cot} x}\right. \right.',
    r'\left( {\mathop{{a}}\nolimits^{{x}}} \left)  \prime =\mathop{{a}}\nolimits^{{x}} \text{ln} a\right. \right.',
    r'\left( {\mathop{{e}}\nolimits^{{x}}} \left)  \prime =\mathop{{e}}\nolimits^{{x}}\right. \right.',
    r'\left( {\mathop{{ \text{log} }}\nolimits_{{a}}x} \left)  \prime =\frac{{1}}{{x \text{ln} a}}\right. \right.',
    r'\left( { \text{ln} a} \left)  \prime =\frac{{1}}{{x}}\right. \right.',
    r'\left( { \text{arcsin} x} \left)  \prime =\frac{{1}}{{\sqrt{{1-\mathop{{x}}\nolimits^{{2}}}}}}\right. \right.',
    r'\left( { \text{arccos} x} \left)  \prime =-\frac{{1}}{{\sqrt{{1-\mathop{{x}}\nolimits^{{2}}}}}}\right. \right.',
    r'\left( { \text{arctan} x} \left)  \prime =\frac{{1}}{{1+\mathop{{x}}\nolimits^{{2}}}}\right. \right.',
    r'\left( { \text{arccot} x} \left)  \prime =-\frac{{1}}{{1+\mathop{{x}}\nolimits^{{2}}}}\right. \right.',
    r' \text{lg} x={\mathop{{ \text{log} }}\nolimits_{{10}}x}',
    r'y={\mathop{{ \text{log} }}\nolimits_{{a}}x{ \left( {a > 0} \right) }}',
    r'\mathop{{a}}\nolimits^{{\mathop{{ \text{log} }}\nolimits_{{a}}x}}=x',
    r'y={\mathop{{ \text{log} }}\nolimits_{{a}}x{ \left( {a > 0} \right) }}',
    r'\mathop{{a}}\nolimits^{{\mathop{{ \text{log} }}\nolimits_{{a}}x}}=x',
    r'\mathop{{ \text{log} }}\nolimits_{{a}}x=\frac{{\mathop{{ \text{log} }}\nolimits_{{b}}x}}{{\mathop{{ \text{log} }}\nolimits_{{b}}a}}',
    r'\mathop{{ \text{log} }}\nolimits_{{a}}b+{\mathop{{ \text{log} }}\nolimits_{{b}}a=1}',
    r'\mathop{{ \text{log} }}\nolimits_{{a}}a=1}',
    r'\mathop{{ \text{log} }}\nolimits_{{a}}1=0}',
    r'\mathop{{ \text{log} }}\nolimits_{{a}}xy={\mathop{{ \text{log} }}\nolimits_{{a}}x+{\mathop{{ \text{log} }}\nolimits_{{a}}y}}',
    r'\mathop{{ \text{log} }}\nolimits_{{a}}\frac{{x}}{{y}}=\mathop{{ \text{log} }}\nolimits_{{a}}x-\mathop{{ \text{log} }}\nolimits_{{a}}y',
    r'\mathop{{ \text{log} }}\nolimits_{{a}}\mathop{{x}}\nolimits^{{ \alpha }}= \alpha {\mathop{{ \text{log} }}\nolimits_{{a}}x}',
    r'f\mathop{{}}\nolimits^{{-1}} \left( f \left( x \left)  \left) =x\right. \right. \right. \right. ',
    r'f \left( f\mathop{{}}\nolimits^{{-1}} \left( x \left)  \left) =x\right. \right. \right. \right. ',
    r'f \left( f\mathop{{}}\nolimits^{{-1}} \left( f \left( x \left)  \left)  \left) =y\right. \right. \right. \right. \right. \right. ',
    r'f \left( f \left( f\mathop{{}}\nolimits^{{-1}} \left( x \left)  \left)  \left) =y\right. \right. \right. \right. \right. \right. ',
    r'\mathop{{ \left( {\frac{{a}}{{b}}} \right) }}\nolimits^{{m}}=\frac{{\mathop{{a}}\nolimits^{{m}}}}{{\mathop{{b}}\nolimits^{{m}}}}',
    r'\mathop{{a}}\nolimits^{{-m}}=\frac{{1}}{{\mathop{{a}}\nolimits^{{m}}}}',
    r'y=\mathop{{x}}\nolimits^{{a}},a \in  \mathbb{R}',
    r'\mathop{{a}}\nolimits^{{m}} \cdot \mathop{{a}}\nolimits^{{n}}=\mathop{{a}}\nolimits^{{m+n}}',
    r'\mathop{{ \left( {\mathop{{a}}\nolimits^{{m}}} \right) }}\nolimits^{{n}}=\mathop{{a}}\nolimits^{{m+n}}',
    r'\frac{{\mathop{{a}}\nolimits^{{m}}}}{{\mathop{{a}}\nolimits^{{n}}}}=\mathop{{a}}\nolimits^{{m-n}}',
    r'\mathop{{a}}\nolimits^{{\frac{{m}}{{n}}}}=\sqrt[{n}]{{\mathop{{a}}\nolimits^{{m}}}}={\mathop{{ \left( {\sqrt[{n}]{{a}}} \right) }}\nolimits^{{m}}}',
    r'\begin{array}{l l}{y=\mathop{{a}}\nolimits^{{x}},{ \left( {a > 0,a \neq 1,x \in  \mathbb{R} } \right) }}\\{y=\mathop{{e}}\nolimits^{{x}}={ \text{exp}  \left( {x} \right) }}\end{array}',
    r'\mathop{{a}}\nolimits^{{0}}=1',
    r'\text{ln} x={\mathop{{ \text{log} }}\nolimits_{{e}}x}',
    r'a > b,b > c \Rightarrow a > c',
    r'\begin{array}{l l}{a > b,c > 0 \Rightarrow ac > bc}\\{a > b,c < 0 \Rightarrow ac < bc}\end{array}',
    r'a > b \Rightarrow a+c > b+c',
    r'a > b,c > d \Rightarrow a+c > b+d',
    r'a > b > 0,c > d > 0 \Rightarrow ac > bd',
    r'a > b > 0,n \in N\mathop{{}}\nolimits^{{ \ast }},n > 1 \Rightarrow a\mathop{{}}\nolimits^{{n}} > b\mathop{{}}\nolimits^{{n}},\sqrt[{n}]{{a}} > \sqrt[{n}]{{b}}',
    r'a\mathop{{}}\nolimits^{{3}}+b\mathop{{}}\nolimits^{{3}}+c\mathop{{}}\nolimits^{{3}} \ge 3abc \left( a > 0,b > 0,c > 0 \right) ',
    r'\left| a \left| - \left| b \left|  \le  \left| a+b \left|  \le  \left| a \left| + \left| b \right| \right. \right. \right. \right. \right. \right. \right. \right. ',
    r'\frac{{2ab}}{{a+b}} \le \sqrt{{ab}} \le \frac{{a+b}}{{2}} \le \sqrt{{\frac{{a\mathop{{}}\nolimits^{{2}}+b\mathop{{}}\nolimits^{{2}}}}{{2}}}} ',
    r'\begin{array}{l l}{\text{当}a > 0,\text{则}}\\{ \left| x \left|  < a \Leftrightarrow x\mathop{{}}\nolimits^{{2}} < a\mathop{{}}\nolimits^{{2}} \Leftrightarrow -a < x < a\right. \right. }\\{ \left| x \left|  > a \Leftrightarrow x\mathop{{}}\nolimits^{{2}} > a\mathop{{}}\nolimits^{{2}} \Leftrightarrow x > a\text{或}x < -a\right. \right. }\end{array}',
    r'\mathop{{H}}\nolimits_{{n}}=\frac{{n}}{{\mathop{ \sum }\limits_{{i=1}}^{{n}}\frac{{1}}{{\mathop{{x}}\nolimits_{{i}}}}}}=\frac{{n}}{{\frac{{1}}{{\mathop{{x}}\nolimits_{{ \text{1} }}}}+\frac{{1}}{{\mathop{{x}}\nolimits_{{2}}}}+ \cdots +\frac{{1}}{{\mathop{{x}}\nolimits_{{n}}}}}}',
    r'\mathop{{G}}\nolimits_{{n}}=\sqrt[{n}]{{\mathop{ \prod }\limits_{{i=1}}^{{n}}\mathop{{x}}\nolimits_{{i}}}}=\sqrt[{n}]{{\mathop{{x}}\nolimits_{{1}}\mathop{{x}}\nolimits_{{2}} \cdots \mathop{{x}}\nolimits_{{n}}}}',
    r'\mathop{{A}}\nolimits_{{n}}=\frac{{1}}{{n}}\mathop{ \sum }\limits_{{i=1}}^{{n}}\mathop{{x}}\nolimits_{{i}}=\frac{{\mathop{{x}}\nolimits_{{1}}+\mathop{{x}}\nolimits_{{2}}+ \cdots +\mathop{{x}}\nolimits_{{n}}}}{{n}}',
    r'\mathop{{Q}}\nolimits_{{n}}=\sqrt{{\mathop{ \sum }\limits_{{i=1}}^{{n}}\mathop{{\mathop{{x}}\nolimits_{{i}}}}\nolimits^{{2}}}}=\sqrt{{\frac{{\mathop{{\mathop{{x}}\nolimits_{{1}}}}\nolimits^{{2}}+\mathop{{\mathop{{x}}\nolimits_{{2}}}}\nolimits^{{2}}+ \cdots +\mathop{{\mathop{{x}}\nolimits_{{n}}}}\nolimits^{{2}}}}{{n}}}}',
    r'\mathop{{H}}\nolimits_{{n}} \le \mathop{{G}}\nolimits_{{n}} \le \mathop{{A}}\nolimits_{{n}} \le \mathop{{Q}}\nolimits_{{n}}',
    r'\mathop{{G}}\nolimits_{{n}}=\sqrt[{n}]{{\mathop{ \prod }\limits_{{i=1}}^{{n}}\mathop{{x}}\nolimits_{{i}}}}=\sqrt[{n}]{{\mathop{{x}}\nolimits_{{1}}\mathop{{x}}\nolimits_{{2}} \cdots \mathop{{x}}\nolimits_{{n}}}}',
    r'\mathop{{Q}}\nolimits_{{n}}=\sqrt{{\frac{{\mathop{ \sum }\limits_{{i=1}}^{{n}}\mathop{{\mathop{{x}}\nolimits_{{i}}}}\nolimits^{{2}}}}{{n}}}}=\sqrt{{\frac{{\mathop{{\mathop{{x}}\nolimits_{{1}}}}\nolimits^{{2}}+\mathop{{\mathop{{x}}\nolimits_{{2}}}}\nolimits^{{2}}+ \cdots +\mathop{{\mathop{{x}}\nolimits_{{n}}}}\nolimits^{{2}}}}{{n}}}}',
    r'\mathop{{A}}\nolimits_{{n}}=\frac{{1}}{{n}}\mathop{ \sum }\limits_{{i=1}}^{{n}}\mathop{{x}}\nolimits_{{i}}=\frac{{\mathop{{x}}\nolimits_{{1}}+\mathop{{x}}\nolimits_{{2}}+ \cdots +\mathop{{x}}\nolimits_{{n}}}}{{n}}',
    r'\mathop{{H}}\nolimits_{{n}}=\frac{{n}}{{\mathop{ \sum }\limits_{{i=1}}^{{n}}\frac{{1}}{{\mathop{{x}}\nolimits_{{i}}}}}}=\frac{{n}}{{\frac{{1}}{{\mathop{{x}}\nolimits_{{ \text{1} }}}}+\frac{{1}}{{\mathop{{x}}\nolimits_{{2}}}}+ \cdots +\frac{{1}}{{\mathop{{x}}\nolimits_{{n}}}}}}',
    r'\begin{array}{l l}{\text{当}x \in  \left( 0,\frac{{ \pi }}{{2}} \left) ,\right. \right. }\\{\text{则}\text{s}\text{i}\text{n}x < x < \text{t}\text{a}\text{n}x}\end{array}',
    r'\begin{array}{l l}{\text{当}x \in  \left( 0,\frac{{ \pi }}{{2}} \left) ,\right. \right. }\\{\text{则}1 < \text{s}\text{i}\text{n}x+\text{c}\text{o}\text{s}x \le \sqrt{{2}}}\end{array}',
    r'\left| \text{s}\text{i}\text{n}x \left| + \left| \text{c}\text{o}\text{s}x \left|  \ge 1\right. \right. \right. \right.'
]

fa_1 = [1,2,3,4,5,6,7,8,9,0]
fa_2 = ['a','b','c','d','x','y','i','u','n','m','p']
fa_3 = ['a','b','x','y','1','2','3','m','n',r'\alpha']
fa_4 = ['cos','sin','tan','sec','csc']

def __replace_content__(s,replace_lists):
    rlists = []
    for l in replace_lists:
        rlists = rlists + l
    r_len = len(re.findall('%s',s))
    r_str = ["\'" + str(rlists[np.random.randint(0,len(rlists))]) + "\'" for idx in range(r_len)]

    # print('input s:', s)
    r_str = '(' + ','.join(r_str) + ')'
    new_str = s % eval(r_str)

    return new_str  

# 采用正则方式匹配
def __replace_content_re__(s, r_lists=[], d_lists=[], o_lists=[]):


    s = re.sub(r'%s', lambda x: str(r_lists[np.random.randint(0,len(r_lists))]), s)
    s = re.sub(r'%d', lambda x: str(d_lists[np.random.randint(0,len(d_lists))]), s)
    s = re.sub(r'%o', lambda x: str(o_lists[np.random.randint(0,len(o_lists))]), s)
    return s

def gen_simple_number_latex(size=999):
    n_lists = list(range(1000))
    n_lists = [str(x) for x in n_lists]
    return n_lists

# 生成简单运算公式
def gen_simple_arithmetic_latex(size=100):
    base_argm = [
        r'%s \times %s = %s \times %s',
        r'%s + %s =%s + %s',
        r'%s - %s =%s - %s',
        r'\frac{ { %s } } { { %s } }',
        r'\frac{{%s}}{{%s}} \times \frac{{%s}}{{%s}}=\frac{{%s}}{{%s}}',
        r'\frac{{%s}}{{%s}} \times %s = \frac{{%s}}{{%s}}',
        r'\frac{{%s}}{{%s}} \div %s = \frac{{%s}}{{%s}}',
        r'\frac{{%s}}{{%s}} \div %s - \frac{{%s}}{{%s}}',
    ]

    sma_lists = []

    for idx in range(size):
        s = __replace_content_re__(base_argm[np.random.randint(0, len(base_argm))], r_lists=['1','2','3','4','5','6','7','8','9','10','x','y','a','b','c'], 
                                                    d_lists=None, 
                                                    o_lists=None)

        sma_lists.append(s)
    print('\n'.join(sma_lists))
    return sma_lists

# 生成一般运算公式
def gen_normal_arithmetic_latex(size=100):
    base_argm = [
        r'\sqrt [{ %s }]{{ %s %s}}=\sqrt[{ %s }]{{ %s }} \cdot \sqrt[{ %s }]{{ %s }}, %s \ge 0, %s \ge 0',
        r'\frac{{ %s }}{{ %s }}=\frac{{ %s }}{{ %s }} \Rightarrow {\frac{{ %s }}{{ %s }}=\frac{{ %s }}{{ %s }}}',
        r'\frac{{ %s }}{{ %s }}=\frac{{ %s }}{{ %s }} \Rightarrow {\frac{{%s-%s}}{{%s}}=\frac{{%s-%s}}{{%s}}}',
        r'\mathop {{ \left( { \frac{{ %s }}{{%s}}} \right) }} \nolimits ^ {{%s}}= \frac {{ \mathop{{%s}} \nolimits ^ {{ %s }}}}{{ \mathop {{ %s }} \nolimits ^ {{%s}}}}',
        r'\sqrt [{ %s }]{{ \frac {{ %s }}{{ %s }}}}= \frac {{ \sqrt [{%s}]{{%s}}}}{{ \sqrt [{ %s }]{{ %s }}}},%s \ge 0,%s \ge 0',
        r'\sqrt [{%s}] {{ \frac {{%s}}{{%s}}}}= \frac {{ \sqrt [{%s}]{{ %s }}}}{{ \sqrt [{%s}]{{%s}}}}{ \left( {%s > 0,%s > 0} \right) }',
        r'\sqrt [{%s}]{{\mathop{{%s}}\nolimits^{{%s}}}}=\mathop{{ \left( {\sqrt[{%s}]{{%s}}} \right) }} \nolimits ^ {{%s}},%s \ge 0',
        r'\left\{ \begin{array} {l l} {x = %s %s} \\ { \frac {{%s}}{{%s}} = \frac {{%s}}{{%s}}} \\ { \frac {{%s}}{{%s}}= \frac {{%s}}{{%s}}} \\ \end{array} \right. ',
        r'\left\{ \begin{array} {l l} {x = %s %s} \\ { %s \times %s = %s } \\ {  %s \div %s = %s } \\ \end{array} \right. '
    ]

    sma_lists = []

    for idx in range(size):

        s = __replace_content__(base_argm[np.random.randint(0, len(base_argm))], replace_lists=[fa_3])    
        sma_lists.append(s)

    print('\n'.join(sma_lists))
    return sma_lists

# 几何相关公式
def gen_geomerty_arithmetic_latex(size=100):
    base_argm = [
        r'\mathop {{ \text{ %s } }} \nolimits ^ {{%d}} \frac {{ \alpha }}{{ %d }} = \frac{{ 1+ \text{ %s }  \alpha }}{{%d}}',
        r'\text{ %s } \frac{{ \alpha }}{{%d}}=\frac{{ \text{%s}  \alpha }}{{1+ \text{%s}  \alpha }}',
        r'\text {%s }  %o + \text{ %s }  %o = %d \text{%s} \frac{{ %o + %o }}{{%d}} \text{%s} \frac{{ %o - %o }}{{%d}}',
        r'\text {%s }  %o - \text{ %s }  %o = %d \text{%s} \frac{{ %o - %o }}{{%d}} \text{%s} \frac{{ %o + %o }}{{%d}}',
        r'%d \text{%s }  %o  \text{ %s }  %o = \text{%s } { \left( { %o - %o } \right) }+ \text{%s} { \left( { %o + %o } \right) }',
        r'\text{%s} { \left( { %o + %o } \left) =- \text{%s}  %o \right. \right. }',
        r'\text{%s} { \left( { %o - %o } \left) =+ \text{%s}  %o \right. \right. }',
        r'\left\{ \begin{array}{l l} {%d = %d \text{%s} %o } \\ {%d = %d \text{%s} %o }  \\ { %d = %d \text{%s} %o } \end{array}\right. '

    ]

    sma_lists = []
    for idx in range(size):
        s = __replace_content_re__(base_argm[np.random.randint(0, len(base_argm))], r_lists=fa_4, 
                                                    d_lists=['1','2','3','n','x','y'], 
                                                    o_lists=[r'\beta',r'\alpha','\pi','A','B','C',r'\theta'])
        sma_lists.append(s)

    print('\n'.join(sma_lists))
    return sma_lists


# 标准方程
def gen_equation_arithmetic_latex(size=100):
    base_argm = [
        r'%d \left( %s-%s \left) \mathop {{}} \nolimits ^ {{ %d }}+ %d \left( %s-%s \left) \mathop{{}} \nolimits ^ {{%d}} = %d r\mathop{{}}\nolimits^{{%d}}\right. \right. \right. \right.',
        r'%s > %s > 0 ,%s > %s > 0 \Rightarrow %s%s %s%s',
        r'%s \mathop {{}} \nolimits^ {{%d}} =-%d %s %s \text{，} \left( p > 0 \right) ',
        r' \frac {{ %s \mathop{{}} \nolimits ^ {{%d}}}} {{%s \mathop{{}} \nolimits^ {{ %d }}}}-\frac{{ %s \mathop {{}} \nolimits ^ {{ %d }}}}{{ %s \mathop{{}}\nolimits^{{%d}}}}=%d',
    ]

    sma_lists = []
    for idx in range(size):
        s = __replace_content_re__(base_argm[np.random.randint(0, len(base_argm))], 
                                                    r_lists=['x','y','a','b','c','d'], 
                                                    d_lists=['1','2','3','n',' '], 
                                                    o_lists=[r'\beta',r'\alpha',r'\pi','A','B','C'])
        sma_lists.append(s)

    print('\n'.join(sma_lists))
    return sma_lists    

# 生成测试文档所需的数学公式
def gen_use_arithmetic_latex(size=100):
    base_argm = [
        r'\left( %s ^ %d - %d \right) + \left( %s ^ %d + %d \right) = %d',
        r'\frac { %d } { \sqrt { %s } }',
        r'\frac { \sqrt { %s } } { %d } '
    ]

    sma_lists = []
    for idx in range(size):
        s = __replace_content_re__(base_argm[np.random.randint(0, len(base_argm))], 
                                                    r_lists=['x','y','a','b'], 
                                                    d_lists=['1','2','3','4'], 
                                                    o_lists=[r'\beta',r'\alpha',r'\pi','A','B','C'])
        sma_lists.append(s)

    print('\n'.join(sma_lists))
    return sma_lists        

def gen_formula_files(data_root, formulas_lists):
    
    # formulas_lists = [latex_add_space(x).strip() for x in formulas_lists]

    with open(os.path.sep.join([data_root,'im2latex_formulas_custom.txt']), 'w', encoding='utf-8') as f:
        formulas = '\n'.join(formulas_lists)
        f.write(formulas)




if __name__ == '__main__':
    # gen_formula_files('D:\\PROJECT_TW\\git\\data\\im2latex\\data')

    # 生成简单运算公式
    # s_list1 = gen_simple_arithmetic_latex(size=1000)

    # 生成一般运算公式
    # s_list2 = gen_normal_arithmetic_latex(size=1000)

    # 生成几何公式
    # s_list3 = gen_geomerty_arithmetic_latex(size=1000)

    # 生成标准方程
    # s_list4 = gen_equation_arithmetic_latex(size=1000)

    # 生成测试所用公式
    # s_list5 = gen_use_arithmetic_latex(size=500)

    # formulas_lists =  s_list1 +  s_list3 + s_list4 + s_list5

    formulas_lists = gen_simple_number_latex()
    print('')

    gen_formula_files(r'D:\PROJECT_TW\git\data\ocr\number', formulas_lists)