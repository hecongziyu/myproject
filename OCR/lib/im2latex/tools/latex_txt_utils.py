import re


def latex_add_space(text):
    # print('text:', text)
    math_flags = []
    for item in re.finditer(r'[ ]{0,}\\[A-Za-z]{2,}', text):
        math_flags.append((item.group(),item.span()))

    if len(math_flags) == 0:
        return ' '.join(list(text.replace(' ','').strip()))

    math_text = []
    math_text.append(' '.join(list(text[0:math_flags[0][1][0]].replace(' ',''))))
    math_text.append(' ' + math_flags[0][0] + ' ')

    if len(math_flags) > 1:
        for idx in range(1, len(math_flags)):
            _last_pos = math_flags[idx-1][1][1]
            _curet_begin_pos = math_flags[idx][1][0]
            math_text.append(' '.join(list(text[_last_pos:_curet_begin_pos].strip().replace(' ',''))))
            math_text.append(' ' + math_flags[idx][0] + ' ')
    else:
        idx = 0

    math_text.append(' '.join(list(text[math_flags[idx][1][1]:].strip().replace(' ',''))))

    math_text = ''.join(math_text)
    math_text = math_text.replace(r'begin { a r r a y }',r'begin{array}')
    math_text = math_text.replace(r'end { a r r a y }',r'end{array}')

    return math_text


# 去空格
def latex_remove_space(text):
    math_flags = []
    for item in re.finditer(r'[ ]{0,}\\[A-Za-z]{1,}[|\\|\(|\)|\{|\}|\[|\]|\.]{0,} ', text):
    # for item in re.finditer(r'\\[A-Za-z|\\|\(|\)|\{|\}|:]{1,} ', text):
        math_flags.append((item.group(),item.span()))

    if len(math_flags) == 0:
        return text.replace(' ','')

    math_text = [] 

    math_text.append(text[0:math_flags[0][1][0]].replace(' ',''))
    math_text.append(math_flags[0][0])

    if len(math_flags) > 1:

        for idx in range(1, len(math_flags)):
            _last_pos = math_flags[idx-1][1][1]
            _curet_begin_pos = math_flags[idx][1][0]
            math_text.append(text[_last_pos:_curet_begin_pos].replace(' ',''))
            math_text.append(math_flags[idx][0])
    else:
        idx = 0

    math_text.append(text[math_flags[idx][1][1]:].replace(' ',''))
    math_text = ''.join(math_text)

    return math_text

if __name__ == '__main__':
    # text = r'\left. \begin{array} { c c c } { S _ { \sigma \sigma ^ { \prime } } \nonumber } \\ { S _ { \Delta } \nonumber } \\ { \tilde { S } _ { \Delta } } \\ \end{array} \right\} \varpi = ( \varpi + 2 ) \left\{ \begin{array} { c c c } { S _ { \sigma \sigma ^ { \prime } } \nonumber } \\ { S _ { \Delta } \nonumber } \\ { \tilde { S } _ { \Delta } } \\ \end{array} \right. .'
    # m_text = latex_remove_space(text)
    text = r'S=\frac{{1}}{{2}}b'
    m_text = latex_add_space(text)
    print('text:', text)
    print('-----------------------')
    print('m text:', m_text)
    print('after :', latex_remove_space(m_text))