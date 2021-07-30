<font face='georgia'>
    
   <h4><strong>What does tf-idf mean?</strong></h4>

   <p>    
Tf-idf stands for <em>term frequency-inverse document frequency</em>, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.
</p>
    
   <p>
One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.
</p>
    
   <p>
Tf-idf can be successfully used for stop-words filtering in various subject fields including text summarization and classification.
</p>
    
</font>

<font face='georgia'>
    <h4><strong>How to Compute:</strong></h4>

Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.

 <ul>
    <li>
<strong>TF:</strong> Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: <br>

$TF(t) = \frac{\text{Number of times term t appears in a document}}{\text{Total number of terms in the document}}.$
</li>
<li>
<strong>IDF:</strong> Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: <br>

$IDF(t) = \log_{e}\frac{\text{Total  number of documents}} {\text{Number of documents with term t in it}}.$
for numerical stabiltiy we will be changing this formula little bit
$IDF(t) = \log_{e}\frac{\text{Total  number of documents}} {\text{Number of documents with term t in it}+1}.$
</li>
</ul>

<br>
<h4><strong>Example</strong></h4>
<p>

Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
</p>
</font>

<font face='georgia'>
    <h4><strong>1. Build a TFIDF Vectorizer & compare its results with Sklearn:</strong></h4>

<ul>
    <li> As a part of this task you will be implementing TFIDF vectorizer on a collection of text documents.</li>
    <br>
    <li> You should compare the results of your own implementation of TFIDF vectorizer with that of sklearns implemenation TFIDF vectorizer.</li>
    <br>
    <li> Sklearn does few more tweaks in the implementation of its version of TFIDF vectorizer, so to replicate the exact results you would need to add following things to your custom implementation of tfidf vectorizer:
       <ol>
        <li> Sklearn has its vocabulary generated from idf sroted in alphabetical order</li>
        <li> Sklearn formula of idf is different from the standard textbook formula. Here the constant <strong>"1"</strong> is added to the numerator and denominator of the idf as if an extra document was seen containing every term in the collection exactly once, which prevents zero divisions.
            
 $IDF(t) = 1+\log_{e}\frac{1\text{ }+\text{ Total  number of documents in collection}} {1+\text{Number of documents with term t in it}}.$
        </li>
        <li> Sklearn applies L2-normalization on its output matrix.</li>
        <li> The final output of sklearn tfidf vectorizer is a sparse matrix.</li>
    </ol>
    <br>
    <li>Steps to approach this task:
    <ol>
        <li> You would have to write both fit and transform methods for your custom implementation of tfidf vectorizer.</li>
        <li> Print out the alphabetically sorted voacb after you fit your data and check if its the same as that of the feature names from sklearn tfidf vectorizer. </li>
        <li> Print out the idf values from your implementation and check if its the same as that of sklearns tfidf vectorizer idf values. </li>
        <li> Once you get your voacb and idf values to be same as that of sklearns implementation of tfidf vectorizer, proceed to the below steps. </li>
        <li> Make sure the output of your implementation is a sparse matrix. Before generating the final output, you need to normalize your sparse matrix using L2 normalization. You can refer to this link https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html </li>
        <li> After completing the above steps, print the output of your custom implementation and compare it with sklearns implementation of tfidf vectorizer.</li>
        <li> To check the output of a single document in your collection of documents,  you can convert the sparse matrix related only to that document into dense matrix and print it.</li>
        </ol>
    </li>
    <br>
   </ul>

  <p> <font color="#e60000"><strong>Note-1: </strong></font> All the necessary outputs of sklearns tfidf vectorizer have been provided as reference in this notebook, you can compare your outputs as mentioned in the above steps, with these outputs.<br>
   <font color="#e60000"><strong>Note-2: </strong></font> The output of your custom implementation and that of sklearns implementation would match only with the collection of document strings provided to you as reference in this notebook. It would not match for strings that contain capital letters or punctuations, etc, because sklearn version of tfidf vectorizer deals with such strings in a different way. To know further details about how sklearn tfidf vectorizer works with such string, you can always refer to its official documentation.<br>
   <font color="#e60000"><strong>Note-3: </strong></font> During this task, it would be helpful for you to debug the code you write with print statements wherever necessary. But when you are finally submitting the assignment, make sure your code is readable and try not to print things which are not part of this task.
    </p>
    
    
    ## SkLearn# Collection of string documents

  corpus = [
      'this is the first document',
      'this document is the second document',
      'and this is the third one',
      'is this the first document',
  ]
    
    
  # Write your code here.
# Make sure its well documented and readble with appropriate comments.
# Compare your results with the above sklearn tfidf vectorizer
# You are not supposed to use any other library apart from the ones given below

from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy
