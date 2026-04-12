# 🔥 Configuração do Firebase — Guia Rápido

## Desenvolvimento Local

1. Baixe o `serviceAccountKey.json` do [Firebase Console](https://console.firebase.google.com/)
   - Acesse **Configurações do Projeto** → **Contas de serviço**
   - Clique em **Gerar nova chave privada**
   - Salve o arquivo como `serviceAccountKey.json` na raiz do projeto

2. O arquivo já está no `.gitignore` — **NUNCA faça commit dele!**

3. A API vai detectar automaticamente e usar o arquivo local

---

## Deploy no Dokploy (Produção)

Como o arquivo JSON não vai para o GitHub (por segurança), use variável de ambiente:

### Passo a passo:

1. **Abra o arquivo localmente:**
   ```bash
   cat serviceAccountKey.json
   ```

2. **Copie TODO o conteúdo** (tem que ser o JSON completo, incluindo as quebras de linha)

3. **No painel do Dokploy:**
   - Vá em **Environment Variables**
   - Adicione uma nova variável:
     - **Nome:** `FIREBASE_SERVICE_ACCOUNT_JSON`
     - **Valor:** Cole o JSON copiado (exatamente como está no arquivo)

4. **Salve e faça o deploy**

---

## Exemplo do JSON

```json
{
  "type": "service_account",
  "project_id": "seu-projeto-id",
  "private_key_id": "abc123def456...",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhki...\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-xxx@seu-projeto.iam.gserviceaccount.com",
  "client_id": "123456789012345678901",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xxx%40seu-projeto.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

---

## Como a API Detecta

A inicialização verifica nesta ordem:

1. **`FIREBASE_SERVICE_ACCOUNT_JSON`** (variável de ambiente) ← Dokploy usa isso
2. **`FIREBASE_SERVICE_ACCOUNT_PATH`** (arquivo local) ← Desenvolvimento usa isso

Se encontrar o JSON via env var, usa ele. Caso contrário, tenta o arquivo.

---

## Verificar Status

Após o deploy, confira se o Firebase foi configurado corretamente:

```bash
curl https://sua-api.dokploy.app/health
```

Resposta esperada:
```json
{
  "status": "healthy",
  "firebase": "ok",
  ...
}
```

Se aparecer `"firebase": "unreachable"` ou `"not_configured"`, revise a variável de ambiente.

---

## Troubleshooting

### Erro: "Firebase service account não configurado"

- **Causa:** Variável `FIREBASE_SERVICE_ACCOUNT_JSON` não foi configurada no Dokploy
- **Solução:** Adicione a variável com o JSON completo

### Erro: "JSON inválido na variável FIREBASE_SERVICE_ACCOUNT_JSON"

- **Causa:** JSON foi colado com escapes incorretos ou truncado
- **Solução:** Copie e cole o JSON novamente, sem modificar nada

### Erro: "Permission denied" ou "Invalid credentials"

- **Causa:** Service account não tem permissões no Firestore
- **Solução:** No Firebase Console, vá em **Firestore Database** → **Rules** e verifique as permissões do service account

---

## Segurança

✅ **Boas práticas:**
- `serviceAccountKey.json` está no `.gitignore`
- Variável de ambiente é privada no Dokploy
- Nunca exponha o JSON em logs ou respostas da API

❌ **Não faça:**
- Commit do arquivo JSON no GitHub
- Compartilhar o JSON em chats/mensagens
- Printar o conteúdo da variável em logs de produção

---

**Dúvidas?** Confira o [README.md](./README.md) completo para mais detalhes sobre variáveis de ambiente.
