fis = readfis('Salario_para_aluno');
figure;
plotfis(fis)
figure;
plotmf(fis,'input',1)
figure;
plotmf(fis,'input',2)
figure;
plotmf(fis,'output',1)
figure;
gensurf(fis)
evalfis([17 13],fis)