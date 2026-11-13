# Diretrizes de Desenvolvimento e Manutenção

Este documento estabelece as regras obrigatórias para desenvolvimento, manutenção e evolução do projeto.

**Todas as alterações realizadas por desenvolvedores ou agentes de IA (Codex) devem respeitar estas diretrizes.**

## 1. Princípios gerais

- os notebooks devem funcionar exatamente como a api
- Priorize soluções simples, legíveis, testáveis e de fácil manutenção.
- Preserve a arquitetura existente, evitando refatorações desnecessárias.
- Não introduza bibliotecas, frameworks ou padrões arquiteturais sem uma justificativa técnica.
- Prefira alterações pequenas e incrementais, com responsabilidades bem definidas.
- Não modifique comportamentos existentes sem validar seus impactos.
- Não implemente funcionalidades fora do escopo solicitado.
- Evite duplicação de código e abstrações prematuras.
- Sempre que possível, reutilize componentes e funções existentes.

## 2. Desenvolvimento orientado a testes (TDD)

**REGRA ABSOLUTA: toda implementação ou alteração de comportamento na API deve seguir obrigatoriamente o ciclo TDD (Red → Green → Refactor).**

### 2.1. RED — Escrever os testes primeiro

Antes de escrever ou modificar o código de produção:

1. Analise os requisitos e identifique o comportamento esperado.
2. Crie testes automatizados que representem esse comportamento.
3. Inclua cenários de sucesso, falhas e casos extremos relevantes.
4. Execute os testes e confirme que falham pelo motivo esperado.

**É proibido implementar a solução antes de criar os testes correspondentes.**

### 2.2. GREEN — Implementar a solução mínima

Após confirmar a falha dos testes:

1. Implemente apenas o código necessário para fazê-los passar.
2. Evite funcionalidades adicionais ou otimizações prematuras.
3. Execute novamente os testes.
4. Confirme que a implementação atende aos requisitos definidos.

### 2.3. REFACTOR — Melhorar sem alterar comportamento

Com os testes passando:

1. Melhore a legibilidade, organização e manutenção do código.
2. Remova duplicações e simplifique estruturas desnecessárias.
3. Preserve o comportamento validado pelos testes.
4. Execute novamente a suíte para identificar possíveis regressões.

### 2.4. Regras obrigatórias

- Nunca pule a etapa RED.
- Nunca crie testes apenas depois de implementar uma funcionalidade.
- Nunca altere testes exclusivamente para fazer uma implementação incorreta passar.
- Nunca remova testes existentes sem justificativa técnica.
- Correções de bugs devem começar por um teste que reproduza o problema.
- Toda nova regra de negócio deve possuir cobertura de testes.
- Se não for possível executar os testes, informe explicitamente a limitação. Nunca declare que passaram sem executá-los.

## 3. Manutenção do motor RAG

### 3.1. Integridade dos dados e índices

- Preserve a ordem dos documentos e seu alinhamento com as linhas da matriz TF-IDF.
- Garanta que cada documento recuperado corresponda aos metadados e identificadores corretos.
- Não altere a estrutura do corpus sem considerar seus impactos nos índices existentes.
- Sempre que uma mudança invalidar o índice, realize sua reconstrução de forma explícita e controlada.
- Não utilize índices incompatíveis com a versão atual do corpus.

### 3.2. Estratégia de recuperação

- **TF-IDF é a estratégia de recuperação padrão do projeto.**
- Não substitua TF-IDF por embeddings, busca vetorial neural ou outras estratégias sem uma decisão explícita.
- Preserve o uso de similaridade de cosseno e os mecanismos existentes de seleção de evidências.
- Alterações no ranking devem ser acompanhadas de testes e avaliação de regressão.
- Não presuma que uma pontuação de similaridade elevada garante uma resposta correta.

### 3.3. Testes do motor

- Não gere ou carregue o corpus completo em testes unitários.
- Utilize fixtures pequenas, determinísticas e representativas.
- Teste separadamente a vetorização, recuperação, ranking e seleção de evidências.
- Inclua cenários com documentos irrelevantes, consultas sem correspondência e múltiplos resultados.
- Evite que testes unitários dependam de serviços externos ou downloads de modelos.

## 4. Responsabilidades da API

A API deve ser responsável por disponibilizar funcionalidades e executar operações de consulta, não por preparar os dados do projeto.

**É proibido executar coleta de dados ou reconstrução de índices durante o processamento de uma requisição HTTP.**

Regras:

- A coleta e a indexação devem ocorrer em processos próprios, fora do fluxo de requisições.
- Os índices devem estar preparados antes de serem utilizados pela API.
- Recursos compartilhados podem ser inicializados uma vez durante a inicialização da aplicação.
- Evite recarregar modelos ou índices a cada requisição.
- Separe as responsabilidades de rotas, regras de negócio, recuperação e geração de respostas.
- Trate falhas de forma explícita, sem mascarar exceções inesperadas.
- Não exponha informações sensíveis em logs ou respostas HTTP.

## 5. Validação obrigatória

Antes de concluir alterações relacionadas ao motor RAG ou à API, execute:

```bash
python -m app.cli verify
pytest
```

Os comandos devem ser executados no ambiente configurado para o projeto.

### Critérios de conclusão

Uma alteração só pode ser considerada concluída quando:

- Os testes relevantes foram escritos antes da implementação.
- O ciclo RED → GREEN → REFACTOR foi respeitado.
- Os testes novos passaram.
- Os testes existentes continuam passando.
- A integridade entre documentos, metadados e matriz TF-IDF foi preservada.
- Não foram introduzidas alterações de escopo não solicitadas.
- Os resultados das verificações foram informados.

Caso alguma validação falhe, registre o problema e não apresente a implementação como totalmente validada.

## 6. Diretrizes para agentes de IA (Codex)

Ao receber uma tarefa:

1. **Entenda:** analise os requisitos, o código existente e os impactos da alteração.
2. **Localize:** identifique os arquivos, funções e testes diretamente relacionados.
3. **Teste primeiro:** escreva os testes necessários e confirme a falha esperada.
4. **Implemente:** realize a menor alteração capaz de atender aos requisitos.
5. **Refatore:** melhore o código sem alterar o comportamento esperado.
6. **Valide:** execute os testes e verificações pertinentes.
7. **Reporte:** apresente um resumo objetivo das alterações realizadas.

### Restrições

- Não reescreva módulos inteiros quando alterações localizadas forem suficientes.
- Não realize migrações de arquitetura por iniciativa própria.
- Não altere contratos públicos da API sem necessidade explícita.
- Não adicione dependências desnecessárias.
- Não ignore erros de testes.
- Não simule resultados de validação.
- Não execute operações externas com custo financeiro sem autorização.

### Formato esperado ao finalizar uma tarefa

Apresente:

**Arquivos alterados:** quais arquivos foram modificados.

**Implementação:** quais comportamentos foram adicionados ou corrigidos.

**TDD:** quais testes foram criados primeiro e qual falha foi observada na etapa RED.

**Validação:** quais comandos foram executados e seus resultados.

**Pendências:** limitações, riscos ou verificações que ainda precisam ser realizadas.

---

## 7. Prioridade das diretrizes

Em caso de conflito entre uma implementação proposta e estas diretrizes:

1. Preserve a integridade dos dados e o funcionamento do sistema.
2. Respeite obrigatoriamente o ciclo TDD nas alterações da API.
3. Mantenha os contratos e a arquitetura existentes.
4. Priorize simplicidade e manutenção.
5. Documente qualquer impedimento técnico.

**Nenhuma conveniência de implementação justifica ignorar as regras obrigatórias deste documento.**
