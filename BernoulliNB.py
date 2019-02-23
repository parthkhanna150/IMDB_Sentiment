import math
class B_NaiveBayes:
    def get_vocab(self, review_list, classes, n_top_words=5000):
        top_words = []
        word_count = {}

        review_str = " ".join(review_list)
        unique_words = list(set(review_str.lower().split()))
        for word in unique_words:
            word_count[word] = 0

        review_list = review_str.lower().split()
        for word in review_list:
            word_count[word] += 1

        word_count.items().sort(key=dict[1], reverse=True)

        for word, count in word_count[0:n_top_words]:
            top_words.append(word)

    def get_vocab_terms_from_doc(V, d):
        ans = []
        doc_list_form = d.lower().split()
        for t in doc_list_form:
            if t in V:
                ans.append(t)
        return ans


    # V = number of terms in the selected vocabulary
    # N = Number of documents
    # Classes = List of classes
    # Docs = List of Documents
    # prior = probability of class c appearing for all docs
    def train_text(docs, Classes):
        prior = []
        cond_prob = [][]
        V = self.get_vocab(docs,Classes)
        N = len(docs)
        for c in Classes:
            N_c = count_docs_c(docs,c)
            prior[c] = N_c/N
            for t in V:
                N_ct = count_docs_c_t(docs,c,t)

                # laplace smoothing
                cond_prob[t][c] = (N_ct + 1)/(N_c + 2)
        return V, prior, cond_prob

    def run_naivebayes(Classes, V, prior, cond_prob, doc):
        V_d = get_vocab_terms_from_doc(V,doc)
        for c in Classes:
            score[c] = log(prior[c])
            for t in V:
                if t in V_d:
                    score[c] += log(cond_prob[t][c])
                else:
                    score[c] += log(1-cond_prob[t][c])
        return score.index(max(score))
