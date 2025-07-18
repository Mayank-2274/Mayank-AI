---
language: en
license: apache-2.0
library_name: transformers
tags:
  - gpt2
  - text-generation
  - medicine
  - india
  - pharmaceutical
  - question-answering
base_model: gpt2
---
# Mayank-AI: Medical AI Assistant Model

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/Mayank-22/Mayank-AI)
[![License](https://img.shields.io/badge/License-Not%20Specified-orange.svg)](https://huggingface.co/Mayank-22/Mayank-AI)
[![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI-red)](https://huggingface.co/Mayank-22/Mayank-AI)

## 📋 Model Overview

Mayank-AI is a specialized artificial intelligence model designed for Indian pharmaceutical and medical applications, trained on comprehensive Indian medicines datasets. This model leverages supervised learning techniques built on GPT-2 transformer architecture to provide accurate and contextually relevant information about Indian medicines, their compounds, uses, and related medical information.

## 🔍 Model Details

### Model Description
- **Developed by:** Mayank Malviya
- **Model Type:** GPT-2 based Transformer for Indian Medical/Pharmaceutical Applications
- **Language(s):** English (with Indian medical terminology and drug names)
- **License:** [More Information Needed]
- **Domain:** Indian Pharmaceuticals & Medicine Information
- **Primary Use:** Indian medicine information, drug compound analysis, symptom mapping, prescription guidance

### Key Features
- ✅ Indian medicines database knowledge
- ✅ Drug compound information and analysis
- ✅ Symptom-to-medicine mapping
- ✅ Prescription guidance and recommendations
- ✅ Disease diagnosis assistance
- ✅ Indian pharmaceutical market insights
- ✅ Medicine availability and alternatives

## 🚀 Quick Start

### Installation
```bash
pip install transformers torch
```

### Basic Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "Mayank-22/Mayank-AI"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example queries for Indian medicines
query1 = "What is the composition of Crocin tablet?"
query2 = "Which medicine is used for fever and headache?"
query3 = "What are the side effects of Paracetamol?"
query4 = "Medicines available for diabetes in India"

# Process query
inputs = tokenizer.encode(query1, return_tensors="pt")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Advanced Usage
```python
# For more controlled generation about Indian medicines
def generate_medicine_response(question, max_length=256):
    prompt = f"Indian Medicine Query: {question}\nResponse:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.6,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Response:")[-1].strip()

# Example usage
question = "What are the uses of Azithromycin tablets available in India?"
answer = generate_medicine_response(question)
print(answer)
```

## 📊 Performance & Capabilities

### Supported Medical Areas
- **Indian Pharmaceuticals:** Comprehensive database of medicines available in India
- **Drug Compounds:** Active ingredients, chemical compositions, formulations
- **Symptom Analysis:** Symptom-to-medicine mapping and recommendations
- **Disease Information:** Common diseases and their standard treatments in India
- **Prescription Guidance:** Dosage, administration, and usage instructions
- **Drug Interactions:** Side effects and contraindications
- **Medicine Alternatives:** Generic and branded medicine alternatives

### Performance Metrics
- **Training Data:** Indian medicines dataset with comprehensive drug information
- **Specialization:** Focused on Indian pharmaceutical market and medicine availability
- **Coverage:** Extensive database of Indian medicines, their compounds, and uses
- **Accuracy:** High precision in Indian medicine information and drug compound details

## ⚠️ Important Medical Disclaimer

> **CRITICAL NOTICE:** This model is for informational and educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

### Limitations & Risks
- **Not a replacement for medical professionals**
- **May contain inaccuracies or outdated information**
- **Should not be used for emergency medical situations**
- **Requires human oversight for clinical applications**
- **May have biases present in training data**

## 🎯 Intended Use Cases

### ✅ Appropriate Uses
- Indian pharmaceutical research and education
- Medicine information lookup and comparison
- Drug compound analysis and research
- Symptom-to-medicine mapping assistance
- Prescription guidance and dosage information
- Medicine availability and alternatives research
- Healthcare app development and integration

### ❌ Inappropriate Uses
- Direct patient diagnosis
- Emergency medical decisions
- Prescription or treatment recommendations without medical supervision
- Replacement for clinical judgment
- Use without proper medical context

## 🔧 Technical Specifications

### Model Architecture
- **Base Architecture:** GPT-2 Transformer model
- **Fine-tuning:** Supervised learning on Indian medicines dataset
- **Context Length:** Standard GPT-2 context window
- **Training Approach:** Domain-specific fine-tuning on pharmaceutical data

### Training Details
- **Training Data:** Indian medicines dataset including:
  - Medicine names and brand information
  - Drug compounds and chemical compositions
  - Symptom-medicine mappings
  - Prescription guidelines and dosages
  - Disease-treatment associations
  - Side effects and contraindications
- **Training Regime:** Supervised fine-tuning on GPT-2 with pharmaceutical domain adaptation
- **Optimization:** Adam optimizer with learning rate scheduling
- **Data Focus:** Indian pharmaceutical market and medicine availability

## 📚 Datasets & Training

### Training Data Sources
- Comprehensive Indian medicines database
- Drug compound and chemical composition data
- Symptom-medicine relationship mappings
- Prescription guidelines and dosage information
- Disease-treatment associations
- Medicine availability and market data

### Data Preprocessing
- Medicine name normalization and standardization
- Drug compound data structure optimization
- Symptom-medicine relationship mapping
- Quality filtering and validation of pharmaceutical data
- Indian market-specific data curation

## 🧪 Evaluation & Validation

### Evaluation Metrics
- **Medicine Information Accuracy:** Correctness of drug compound and usage information
- **Symptom Mapping Precision:** Accuracy of symptom-to-medicine recommendations
- **Indian Market Relevance:** Appropriateness for Indian pharmaceutical context
- **Safety Assessment:** Risk evaluation for medicine information provision

### Benchmark Performance
- **Indian Medicine Database:** Comprehensive coverage of medicines available in India
- **Drug Compound Accuracy:** High precision in chemical composition information
- **Symptom-Medicine Mapping:** Effective symptom-to-treatment recommendations

## 🔄 Updates & Maintenance

This model is maintained and updated with:
- Latest Indian medicine information
- New drug approvals and market entries
- Updated compound and formulation data
- Enhanced symptom-medicine mappings

## 📖 Citation

If you use this model in your research, please cite:

```bibtex
@misc{mayank2024indianmedicines,
  title={Mayank-AI: Indian Medicines Information Model},
  author={Malviya, Mayank},
  year={2024},
  url={https://huggingface.co/Mayank-22/Mayank-AI},
  note={GPT-2 based model for Indian pharmaceutical information}
}
```

## 🤝 Contributing

Contributions to improve the model are welcome! Please:
- Report issues with medicine information accuracy
- Suggest new Indian medicines to include
- Share feedback on drug compound data
- Contribute to symptom-medicine mapping improvements

## 📞 Contact & Support

- **Model Author:** Mayank Malviya
- **Repository:** [Mayank-22/Mayank-AI](https://huggingface.co/Mayank-22/Mayank-AI)
- **Issues:** Please report issues through the Hugging Face repository

## 📄 License

This model's license is not currently specified. Please check the repository or contact the author for licensing information.

## 🙏 Acknowledgments

Special thanks to the Indian pharmaceutical community, healthcare professionals, and medical researchers who contributed to the development and validation of this specialized model for Indian medicines.

---

**Remember:** This AI model is a tool to assist, not replace, medical professionals. Always prioritize patient safety and seek professional medical advice for healthcare decisions.
