from utils.token_utils import lexer

if __name__ == '__main__':
    data = '1——3 ABC （4）D（5）B（6）C（7）A（8）C（9）B（10）D'
    data = '（1）0（2）5（3）2（14）[{img:255}] （15）{img:256}（16）{img:257}（17）(0,±1) 4.0'
    lexer = lexer()

    lexer.input(data)

    # Tokenize
    while True:
        tok = lexer.token()
        if not tok: 
            break      # No more input
        print(tok)
        print(tok.__dict__)


    print(data[4:])