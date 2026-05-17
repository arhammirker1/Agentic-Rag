from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import styles
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import enums

stylesheets = getSampleStyleSheet()
title = stylesheets["Title"]
body = stylesheets["BodyText"]
body.fontSize = 10
body.leading = 14

docs = {
"doc1_ai_agent_foundations.pdf": ("AI Agent Foundations and Evolution",[
("Origins of Intelligent Systems",
"""Artificial intelligence evolved from symbolic systems toward learning-based systems. Early AI systems relied on explicit logic and rules, but modern systems increasingly depend on neural architectures. A central turning point emerged through attention mechanisms and transformer architectures. Researchers discovered that systems could model relationships across large sequences and preserve context across tasks.

The evolution of AI agents created a distinction between models and agents. A model predicts outputs. An agent observes environments, stores state, plans actions, and executes tool usage. This distinction became increasingly important with large language models and autonomous workflows.

Repeated concepts: transformers, attention, memory systems, tool use, retrieval augmented generation, computer use, and planning."""*8),
("Memory and Tool Use",
"""Memory transformed agents from one-step systems into persistent systems. Long-term memory architectures, retrieval systems, episodic memory, semantic memory, and external knowledge stores became important.

Tool use emerged as another capability. Agents increasingly interacted with browsers, APIs, spreadsheets, and databases. Tool execution introduced planning loops: observe, reason, act, verify, and repeat.

Relationships likely appear with retrieval systems, multimodal agents, and computer-use systems discussed in other documents."""*8),
("Architectural Patterns",
"""Modern AI agent patterns include planner-executor models, reflection loops, hierarchical agents, and graph-based execution systems. Multi-agent coordination introduces communication protocols and dependency structures."""*10)
]),

"doc2_multimodal_computer_use.pdf":("Multimodal Systems and Computer-Use Agents",[
("From Text to Action",
"""Multimodal systems integrate text, image, audio, interface signals, and external actions. Instead of generating text alone, these systems manipulate software environments and computers.

Computer-use agents frequently rely on screenshots, cursor position, planning loops, and environmental observations. These systems connect naturally with memory and retrieval systems from earlier discussions."""*9),
("Human Computer Interaction",
"""Researchers increasingly explore desktop agents capable of controlling keyboards and mice. Systems analyze interfaces and infer intent. Attention mechanisms again play central roles because interfaces contain structured visual information.

Repeated entities: transformers, memory systems, planning agents, retrieval pipelines, and tool use."""*10),
("Failure Modes",
"""Computer-use systems suffer from hallucinated interface states, incorrect assumptions, and unstable planning. Reflection systems attempt to solve these issues through iterative reasoning."""*10)
]),

"doc3_vectorless_rag_graphs.pdf":("Vectorless Retrieval and Graph Knowledge Systems",[
("Traditional Retrieval",
"""Retrieval Augmented Generation traditionally uses vector embeddings. Embeddings map documents into dense representations. Similarity search then identifies related information.

However, vector systems sometimes fail when semantic relationships are subtle or when explicit entity relationships matter."""*9),
("Vectorless Graph Retrieval",
"""Vectorless retrieval introduces entity extraction, graph construction, relationship mapping, and symbolic navigation. Documents become nodes and concepts become edges.

Examples include relationships between attention mechanisms, tool use, multimodal systems, AI agents, and memory architectures. Systems can generate relationship trees connecting entities across many documents."""*11),
("Knowledge Graph Patterns",
"""Graph systems frequently support document exploration. Nodes can represent concepts, companies, people, or technical artifacts. Traversal enables explanation pathways."""*10)
]),

"doc4_ai_startup_case_studies.pdf":("AI Startup Systems and Case Studies",[
("Operational AI Systems",
"""Modern startups increasingly deploy AI agents in production. Applications include marketing automation, campaign generation, autonomous analytics, customer support systems, and document processing.

Organizations combine retrieval systems, graph architectures, tool use, memory layers, and planning frameworks."""*10),
("Autonomous Workflows",
"""Workflow systems involve chains of execution. Data arrives from APIs, databases, email systems, and user interactions. Agents determine actions through planning and reflection loops.

Repeated relationships include retrieval augmented generation, vectorless systems, memory stores, and multimodal interfaces."""*10),
("Future Directions",
"""Future AI systems may increasingly resemble operating systems rather than chat interfaces. Systems coordinate memory, planning, perception, and action."""*10)
])
}

for filename,(doctitle,sections) in docs.items():
    doc=SimpleDocTemplate("./testpdf/"+filename)
    story=[Paragraph(doctitle,title),Spacer(1,12)]
    for head,text in sections:
        h=stylesheets["Heading2"]
        story.append(Paragraph(head,h))
        for p in text.split("\n\n"):
            story.append(Paragraph(p,body))
            story.append(Spacer(1,6))
        story.append(Spacer(1,12))
    doc.build(story)

print("done")
