bool Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    json data = json::parse(f);
    
    json model_vocab;
    if (data.contains("model") && data["model"].contains("vocab")) {
        model_vocab = data["model"]["vocab"];
    } else if (data.contains("vocab")) {
        model_vocab = data["vocab"];
    } else {
        return false;
    }
    
    vocab.clear();
    int max_id = 0;
    for (auto& el : model_vocab.items()) max_id = std::max(max_id, (int)el.value());
    inv_vocab.assign(max_id + 1, "");

    for (auto& el : model_vocab.items()) {
        vocab[el.key()] = el.value();
        inv_vocab[el.value()] = el.key();
    }
    return true;
}