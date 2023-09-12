from pinecone_datasets.dataset import Dataset

sentences = [
   "How do I get a replacement Medicare card?",
   "What is the monthly premium for Medicare Part B?",
   "How do I terminate my Medicare Part B (medical insurance)?",
   "How do I sign up for Medicare?",
   "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
   "How do I sign up for Medicare Part B if I already have Part A?",
   "What are Medicare late enrollment penalties?",
   "What is Medicare and who can get it?",
   "How can I get help with my Medicare Part A and Part B premiums?",
   "What are the different parts of Medicare?",
   "Will my Medicare premiums be higher because of my higher income?",
   "What is TRICARE ?",
   "Should I sign up for Medicare Part B if I have Veterans' Benefits?"
]

def test_sanity():
   dataset = Dataset.from_sentence_transformers(
      'sentence-transformers/all-MiniLM-L6-v2',
      sentences
   )

   head = dataset.head()
   
   print('head')
   
   assert(dataset.documents.shape[0]==len(sentences))