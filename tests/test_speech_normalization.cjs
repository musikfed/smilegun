function normalizeSpeech(text){ return String(text||"").toLowerCase().replace(/ё/g,"е").replace(/[^a-zа-я]+/g," ").trim(); }
function editDistance(a,b){
  const m=a.length,n=b.length,dp=Array.from({length:m+1},()=>Array(n+1).fill(0));
  for(let i=0;i<=m;i++)dp[i][0]=i; for(let j=0;j<=n;j++)dp[0][j]=j;
  for(let i=1;i<=m;i++)for(let j=1;j<=n;j++)dp[i][j]=Math.min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1));
  return dp[m][n];
}
function speechCommand(text){
  const s=normalizeSpeech(text); if(!s)return null;
  const fireWords=new Set(["пиу","пью","пию","пэу","пеу","пю","пьюу","pew"]);
  const chargeWords=new Set(["пым","пим","пум","пэм","пем","бым","бим","бум","дым","тым","пын","пин","пун","pym"]);
  for(const t of s.split(/\s+/).filter(Boolean)){
    if(fireWords.has(t))return "fire";
    if(chargeWords.has(t))return "charge";
    if(t.length>=2&&t.length<=4){ if(editDistance(t,"пиу")<=1)return "fire"; if(editDistance(t,"пым")<=1)return "charge"; }
  }
  return null;
}
const assert=require("node:assert/strict");
for(const w of ["пиу","ПЬЮ","пию","пеу"]) assert.equal(speechCommand(w),"fire",w);
for(const w of ["пым","пим","пум","бум","дым","пын"]) assert.equal(speechCommand(w),"charge",w);
assert.equal(speechCommand("привет"),null);
assert.equal(speechCommand("хорошо"),null);
console.log("speech grammar: ok");
