import pandas as pd
import random

# Specialties and typical keywords
specialties = {
    'Cardiology': ['heart', 'rhythm', 'arrhythmia', 'myocardial', 'infarction', 'chest pain', 'echocardiogram', 'valve', 'murmur', 'hypertension', 'atrial fibrillation', 'coronary', 'artery', 'stent', 'angioplasty', 'tachycardia', 'bradycardia'],
    'Neurology': ['brain', 'nerve', 'seizure', 'stroke', 'epilepsy', 'headache', 'migraine', 'neuropathy', 'mri', 'eeg', 'dementia', 'alzheimer', 'tremor', 'parkinson', 'reflexes', 'cranial', 'spinal'],
    'Orthopedics': ['bone', 'fracture', 'joint', 'ligament', 'tendon', 'arthritis', 'osteoarthritis', 'spine', 'knee', 'hip', 'shoulder', 'acl', 'meniscus', 'surgery', 'cast', 'rehabilitation', 'sprain'],
    'Radiology': ['x-ray', 'ct scan', 'mri', 'ultrasound', 'radiograph', 'imaging', 'contrast', 'lesion', 'tumor', 'cyst', 'fracture', 'opacity', 'density', 'shadow', 'echogenic', 'mammogram'],
    'Dermatology': ['skin', 'rash', 'lesion', 'melanoma', 'eczema', 'psoriasis', 'acne', 'biopsy', 'epidermis', 'dermis', 'erythema', 'pruritus', 'mole', 'carcinoma', 'dermatitis', 'cyst', 'ulcer']
}

# Template sentences
templates = [
    "Patient presents with {word1} and {word2}.",
    "The {word1} indicates possible {word2} issues.",
    "Examination of the {word1} reveals {word2}.",
    "History of {word1} requires monitoring for {word2}.",
    "The procedure involved checking the {word1} for {word2}.",
    "Recommend further tests for {word1} and {word2}.",
    "Symptoms include severe {word1} and mild {word2}.",
    "Diagnosis confirmed {word1} related to {word2}.",
    "Treatment plan includes addressing the {word1} and {word2}.",
    "Follow-up for {word1} showed improvement in {word2}."
]

def generate_report(specialty):
    words = specialties[specialty]
    report_sentences = []
    num_sentences = random.randint(3, 6)
    for _ in range(num_sentences):
        word1, word2 = random.sample(words, 2)
        template = random.choice(templates)
        report_sentences.append(template.format(word1=word1, word2=word2))
    
    # Add some common medical fluff
    fluff = ["Patient is stable.", "Vitals are normal.", "Continue current medications.", "Follow up in 2 weeks."]
    report_sentences.append(random.choice(fluff))
    random.shuffle(report_sentences)
    return " ".join(report_sentences)

def generate_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        specialty = random.choice(list(specialties.keys()))
        report = generate_report(specialty)
        data.append({'medical_specialty': specialty, 'transcription': report})
    
    df = pd.DataFrame(data)
    # Add some noise/imbalance if we want, but let's keep it mostly balanced for now
    return df

if __name__ == '__main__':
    print("Generating synthetic medical transcriptions dataset...")
    df = generate_dataset(2000)
    df.to_csv('medical_transcriptions.csv', index=False)
    print("Dataset generated and saved to medical_transcriptions.csv")
    print(df['specialty'].value_counts())
    print("\nSample Report:")
    print(df['transcription'].iloc[0])
