from make_vocab import load_vocab, load_data, tokenize, encode

def oov_list(sequence, stoi):
    total, oov, seq_with_oov = 0, 0, 0
    for toks in sequence:
        n = sum(t not in stoi for t in toks)
        total += len(toks)
        oov += n
        seq_with_oov += (n > 0)
    return oov / total, seq_with_oov / len(sequence)

def main():
    val_data = load_data("valid")
    test_data = load_data("test")
    _, words_stoi = load_vocab("words")
    _, actions_stoi = load_vocab("actions")
    val_words_seq = []
    test_words_seq = []
    val_actions_seq = []
    test_actions_seq = []

    for line in val_data:
        for sent in line["steps"]:
            val_words_seq.append(tokenize(sent))

    for line in test_data:
        for sent in line["steps"]:
            test_words_seq.append(tokenize(sent))

    val_words_oov, val_words_swo = oov_list(val_words_seq, words_stoi)
    test_words_oov, test_words_swo = oov_list(test_words_seq, words_stoi)

    for line in val_data:
        val_actions_seq.append(line["actions"])
    for line in test_data:
        test_actions_seq.append(line["actions"])

    val_actions_oov, val_actions_swo = oov_list(val_actions_seq, actions_stoi)
    test_actions_oov, test_actions_swo = oov_list(test_actions_seq, actions_stoi)

    print("\n")
    print("Valid(original: valid_seen)")
    print(50*"=")
    print(f"[words oov] : {val_words_oov:.2%}, [sequence with oov] : {val_words_swo:.2%}")
    print(50*"=")
    print(f"[actions oov] : {val_actions_oov:.2%}, [samples with oov] : {val_actions_swo:.2%}")
    print("\n")
    print("test(original: valid_unseen)")
    print(50*"=")
    print(f"[words oov] : {test_words_oov:.2%}, [sequence with oov] : {test_words_swo:.2%}")
    print(50*"=")
    print(f"[actions oov] : {test_actions_oov:.2%}, [samples with oov] : {test_actions_swo:.2%}")
    print("\n")
    print("[words vocab size] :", len(words_stoi))
    print("[actions vocab size] :", len(actions_stoi))
    print("\n")

    sent1= "Take right, take another right, walk to chair"
    sent2 = "Go right and then right again and turn to the right to face the toilet."
    sent3 = "Carry the remote to the floor lamp that is to the left of the bright red chair"
    list = [sent1, sent2, sent3]

    for sent in list:
        tokens = tokenize(sent)
        ids = encode(tokens, words_stoi)
        print(f"sentence : {sent}")
        print(f"tokens : {tokens}")
        print(f"ID : {ids}")
        print("\n")


if __name__ == "__main__":
    main()