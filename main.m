clc;
clear all, close all;
start0 = tic;
addpath('./eval/');
addpath('../dataOld/');
addpath('./measureFSMSC/'); % FSMSC的指标结果

epsilon = 10^-3;
maxIter = 100;
totalCycle = 5; % 聚类，重复运行的次数，而非五折交叉验证

dataSetName = 'austra';
load(dataSetName);
method = 'Fifth 1.2.5.1';  
% 解法二，μ分为μ1和μ2，迭代中μ为非固定，ρ=3，进出均做归一化
% 参照GMC-LRSSC的解法
X = austra(:,1:14); % X 是 n*d
% X = zscore(X');
label = austra(:,end); % label是n*1
totalClass = numel(unique(label));

% 在report文件夹下保存结果
postName = strcat(dataSetName , '.txt');
saveFileName = strcat('.\report\',method,'_', postName);
warning off;
fid=fopen(saveFileName,'a');    %保存到.\report\wine_v.txt中
fprintf('Method:%s, Dataset:%s, Time:%s\n', method, dataSetName, datetime);
fprintf(fid, 'Method:%s, Dataset:%s, Time:%s\n', method, dataSetName, datetime);
string = 'Fifth 1.2.5.1\n';
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
                    result = []; % 每个参数巡检里，先把大表格清空
                    % 在每个参数中cycle多次
                    for i = 1 : totalCycle % 重复运行
                        start1 = tic;
                        [B, A, Z] = train(zscore(X'), totalClass, alpha, ep1, ep2, epsilon, maxIter);
                        % zscore 或 normalize做列归一化，可能可以提高些许
                        timePerTrain = toc(start1);
                        
                        Z = gather(Z); % 把Z变为CPU数据，否则后面出错
                        % 用Z的svd分解来判定最终结果，来源于FSMSC（TIP2023）
                        % 通过对 Z ∗ 进行 SVD 得到谱嵌入 H，通过对 H 进行 k 均值聚类得到聚类结果。
                        [UU, ~, ~]=svd(Z', 'econ');
                        [res, labelPredict] = myNMIACCwithmean(UU,label,totalClass); 
                        % [ACC nmi Purity Fscore Precision Recall AR Entropy]


                        % [res] = my_eval_y1(labelPredict, label); % res是 [CE; acc; nmi; purity; AR; RI; MI; HI; fscore; precision; recall];
                        result = [result; [res(8) res(1) res(2) res(3) timePerTrain]]; % result 是一个大表格
                    end
                    result(totalCycle+1, :) = mean(result);  % 5轮交叉验证，第6行纪录其均值：准确率，时间
                    result(totalCycle+2, :) = std(result(1:totalCycle, :)) ; % 第7行纪录其方差：准确率，时间
                    result(totalCycle+3, :) = max(result); % 第8行记录最高值：准确率，时间
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
                        paraBest = [alpha, ep1, ep2]; % 不同方法调整
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