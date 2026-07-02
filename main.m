clc;
clear all, close all;
start0 = tic;
addpath('./eval/');
addpath('../dataOld/');
addpath('./measureFSMSC/'); 

epsilon = 10^-3;
maxIter = 100;
totalCycle = 5; 

dataSetName = 'austra';
load(dataSetName);
method = '';  


X = austra(:,1:14);
% X = zscore(X');
label = austra(:,end); 
totalClass = numel(unique(label));


postName = strcat(dataSetName , '.txt');
saveFileName = strcat('.\report\',method,'_', postName);
warning off;
fid=fopen(saveFileName,'a');  
fprintf('Method:%s, Dataset:%s, Time:%s\n', method, dataSetName, datetime);
fprintf(fid, 'Method:%s, Dataset:%s, Time:%s\n', method, dataSetName, datetime);
string = '';
fprintf(string);
fprintf(fid, string);

para = [10^-2; 10^-1; 10^0; 10^1; 10^2];
EP = [10^2; 10^0; 10^-2; 10^-4; 10^-6];
result = [];
paraString = 'alpha:%.4f, ep1:%.6f, ep2:%.6f';

[N,D] = size(X);
minCE = 100;
bestAcc = 0;
bestNmi = 0;
bestPurity = 0;
bestTimePerTrain = 100000;

        for ithAlpha = 1:size(para)
            for ithEp1 = 1:size(EP)
                for ithEp2 = 1:size(EP)
                    alpha = para(ithAlpha);
                    ep1 = EP(ithEp1);
                    ep2 = EP(ithEp2);
                    result = []; 
                    
                    for i = 1 : totalCycle 
                        start1 = tic;
                        [B, A, Z] = train(zscore(X'), totalClass, alpha, ep1, ep2, epsilon, maxIter);
                       
                        timePerTrain = toc(start1);
                        
                        Z = gather(Z); 
                        
                        [UU, ~, ~]=svd(Z', 'econ');
                        [res, labelPredict] = myNMIACCwithmean(UU,label,totalClass); 
                        % [ACC nmi Purity Fscore Precision Recall AR Entropy]


                        % [res] = my_eval_y1(labelPredict, label); % res是 [CE; acc; nmi; purity; AR; RI; MI; HI; fscore; precision; recall];
                        result = [result; [res(8) res(1) res(2) res(3) timePerTrain]]; 
                    end
                    result(totalCycle+1, :) = mean(result);  
                    result(totalCycle+2, :) = std(result(1:totalCycle, :)) ; 
                    result(totalCycle+3, :) = max(result); 
                    avgCE = result(totalCycle+1 , 1);
                    avgAcc = result(totalCycle+1 , 2);
                    avgNmi = result(totalCycle+1 , 3);
                    avgPurity = result(totalCycle+1 , 4);
                    avgTimePerTrain = result(totalCycle+1, 5);
                    stdCE = result(totalCycle+2 , 1);
                    stdAcc = result(totalCycle+2 , 2);
                    stdNmi = result(totalCycle+2 , 3);
                    stdPurity = result(totalCycle+2 , 4);
                    if avgAcc > bestAcc
                        paraBest = [alpha, ep1, ep2]; 
                        fprintf(strcat('CE:%.4f±%.4f ||',paraString,' \n'),avgCE, stdCE, paraBest);
                        fprintf(fid,strcat('CE:%.4f±%.4f ||',paraString,'  \n'),avgCE, stdCE, paraBest);
                        fprintf(strcat('acc:%.4f±%.4f ||',paraString,'  \n'),avgAcc, stdAcc, paraBest);
                        fprintf(fid,strcat('acc:%.4f±%.4f ||',paraString,'  \n'),avgAcc, stdAcc, paraBest);
                        fprintf(strcat('nmi:%.4f±%.4f ||',paraString,'  \n'),avgNmi, stdNmi, paraBest);
                        fprintf(fid,strcat('nmi:%.4f±%.4f ||',paraString,'  \n'),avgNmi, stdNmi, paraBest);
                        fprintf(strcat('purity:%.4f±%.4f ||',paraString,'  \n'),avgPurity, stdPurity, paraBest);
                        fprintf(fid,strcat('purity:%.4f±%.4f ||',paraString,'  \n'),avgPurity, stdPurity, paraBest);
                        

                        minCE = avgCE;
                        minStdCE = stdCE;
                        bestAcc = avgAcc;
                        bestStdAcc = stdAcc;
                        bestNmi = avgNmi;
                        bestStdNmi = stdNmi;
                        bestPurity = avgPurity;
                        bestStdPurity = stdPurity;

                        bestZ = Z;
                        bestB = B;
                        bestA = A;
                        bestPredict = labelPredict;
                    end
        
                    if avgCE < minCE
                        fprintf(strcat('CE:%.4f±%.4f ||',paraString,'  \n'),avgCE, stdCE, paraBest);
                        fprintf(fid,strcat('CE:%.4f±%.4f ||',paraString,'  \n'),avgCE, stdCE, paraBest);
                    end
        
                    if avgNmi > bestNmi
                        fprintf(strcat('nmi:%.4f±%.4f ||',paraString,'  \n'),avgNmi, stdNmi, paraBest);
                        fprintf(fid,strcat('nmi:%.4f±%.4f ||',paraString,'  \n'),avgNmi, stdNmi, paraBest);
                    end
                    
                    if avgPurity > bestPurity
                        fprintf(strcat('purity:%.4f±%.4f ||',paraString,'  \n'),avgPurity, stdPurity);
                        fprintf(fid,strcat('purity:%.4f±%.4f ||',paraString,'  \n'),avgPurity, stdPurity);
                    end
        
                    if avgTimePerTrain < bestTimePerTrain
                        bestTimePerTrain = avgTimePerTrain;
                    end
                end
            end
        end

fprintf('---------------------------------------------------------\n');
fprintf(fid,'---------------------------------------------------------\n');
fprintf(strcat('CE:%.4f±%.4f,acc:%.4f±%.4f,nmi:%.4f±%.4f,purity:%.4f±%.4f,timeTrain:%.4f ||',paraString,'  \n'),minCE,stdCE,bestAcc,bestStdAcc,bestNmi,bestStdNmi,bestPurity,bestStdPurity,bestTimePerTrain,paraBest);
fprintf(fid,strcat('CE:%.4f±%.4f,acc:%.4f±%.4f,nmi:%.4f±%.4f,purity:%.4f±%.4f,timeTrain:%.4f ||',paraString,'  \n'),minCE,stdCE,bestAcc,bestStdAcc,bestNmi,bestStdNmi,bestPurity,bestStdPurity,bestTimePerTrain,paraBest);
fclose(fid);
figure(1);
imagesc(bestZ);
allTime = toc(start0)
%{
figure (1);
data = X*bestB;
scatter(data(:,1),data(:,2),15,bestPredict,'filled');
%}
