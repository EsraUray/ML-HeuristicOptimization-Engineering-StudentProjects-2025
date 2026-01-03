clear; clc; close all;

%% 1. SİSTEM GİRİŞLERİ
P_active = input('1. Mevcut Aktif Güç (kW): ');
Q_inductive_load = input('2. Mevcut Endüktif Reaktif Yük (kVAr): ');

disp('-------------------------------------------------------');
disp('Örnek: [1 1.5 2.5 5 5 10 12.5 15 20 25 50 60]');
CapacitorSteps = input('3. Kondansatör Kademelerini Giriniz [kVAr]: ');

if isempty(CapacitorSteps)
    CapacitorSteps = [1 1.5 2.5 5 5 10 12.5 15 20 25 50 60];
    disp('-> Varsayılan kademeler yüklendi.');
end
CapacitorSteps = sort(CapacitorSteps);

% [YASAL SINIRLAR]
Limit_Inductive = 0.20;  % %20 Endüktif
Limit_Capacitive = 0.15; % %15 Kapasitif

%% 2. ALGORİTMA PARAMETRELERİ
N = length(CapacitorSteps); 
HMS = 60;           
HMCR = 0.98;        
PAR = 0.45;      
MaxIter = 10000;    

%% 3. BAŞLANGIÇ
HM = randi([0, 1], HMS, N); 
Fitness = zeros(HMS, 1);

for i = 1:HMS
    Fitness(i) = CostFunction(HM(i,:), CapacitorSteps, P_active, Q_inductive_load, Limit_Inductive, Limit_Capacitive);
end

[BestCost, BestIdx] = min(Fitness);
[WorstCost, WorstIdx] = max(Fitness);
BestSol = HM(BestIdx, :);
ConvergenceHistory = zeros(MaxIter, 1);

%% 4. OPTİMİZASYON DÖNGÜSÜ
fprintf('\nOptimizasyon ve Bölge Analizi yapılıyor...\n');

for iter = 1:MaxIter
    NewHarmony = zeros(1, N);
    for j = 1:N
        if rand < HMCR
            RandIdx = randi(HMS);
            Value = HM(RandIdx, j);
            if rand < PAR
                NewHarmony(j) = 1 - Value; 
            else
                NewHarmony(j) = Value;
            end
        else
            NewHarmony(j) = randi([0, 1]);
        end
    end
    
    NewCost = CostFunction(NewHarmony, CapacitorSteps, P_active, Q_inductive_load, Limit_Inductive, Limit_Capacitive);
    
    if NewCost < WorstCost
        HM(WorstIdx, :) = NewHarmony;
        Fitness(WorstIdx) = NewCost;
        [BestCost, BestIdx] = min(Fitness);
        [WorstCost, WorstIdx] = max(Fitness);
        BestSol = HM(BestIdx, :);
    end
    ConvergenceHistory(iter) = BestCost;
end

%% 5. SONUÇ HESAPLAMALARI
SelectedSteps = BestSol .* CapacitorSteps;
Q_total_cap = sum(SelectedSteps);
Q_net_error = Q_inductive_load - Q_total_cap;
CurrentRatio = abs(Q_net_error) / P_active;

% Grafik için Y ekseni değeri (Endüktif pozitif, Kapasitif negatif)
if Q_net_error > 0
    Y_Plot_Val = CurrentRatio; % Endüktif Bölge
    SystemStatus = 'ENDÜKTİF (Çekiyor)';
else
    Y_Plot_Val = -CurrentRatio; % Kapasitif Bölge
    SystemStatus = 'KAPASİTİF (Basıyor)';
end

%% 6. RAPORLAMA
disp(' ');
disp('================================================================');
disp('             SONUÇ RAPORU                                       ');
disp('================================================================');
fprintf('Hedef Yük (Q_L) : %.2f kVAr\n', Q_inductive_load);
fprintf('Kompanze (Q_C)  : %.2f kVAr\n', Q_total_cap);
fprintf('Net Hata        : %.2f kVAr [%s]\n', abs(Q_net_error), SystemStatus);
fprintf('Reaktif Oran    : %% %.2f\n', CurrentRatio * 100);

disp('----------------------------------------------------------------');
disp('Seçilen Kondansatörler (Küçük Öncelikli):');
for k = 1:N
    if BestSol(k) == 1
        if CapacitorSteps(k) <= mean(CapacitorSteps)
            Note = '(Hassas)';
        else
            Note = '(Kaba)';
        end
        fprintf('  %2d. Kademe: %6.2f kVAr [AÇIK] %s\n', k, CapacitorSteps(k), Note);
    end
end
disp('================================================================');

%% 7. GRAFİKSEL GÖSTERİM (RENKLİ BÖLGELER)
figure('Name', 'Kompanzasyon Bölge Analizi', 'Color', 'white');

% --- Üst Grafik: İterasyon ---
subplot(2,1,1);
plot(ConvergenceHistory, 'LineWidth', 2, 'Color', [0 0.45 0.74]);
title('Optimizasyon Süreci (Maliyet Azalımı)');
xlabel('İterasyon'); ylabel('Maliyet');
grid on;

% --- Alt Grafik: Ceza Sınırları ---
subplot(2,1,2); hold on;
title(['Sistem Durumu: %' num2str(CurrentRatio*100, '%.2f') ' (' SystemStatus ')']);

% 1. Bölgeleri Çiz (Fill komutu ile)
% Endüktif Ceza Bölgesi (> %20) - Turuncu
fill([0 1 1 0], [Limit_Inductive Limit_Inductive 1 1], [0.9 0.6 0.4], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
% Kapasitif Ceza Bölgesi (> %15) - Kırmızı/Bordo
fill([0 1 1 0], [-Limit_Capacitive -Limit_Capacitive -1 -1], [0.8 0.3 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
% Güvenli Bölge (Arası) - Yeşil/Bej
fill([0 1 1 0], [-Limit_Capacitive -Limit_Capacitive Limit_Inductive Limit_Inductive], [0.8 0.9 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% 2. Sınır Çizgileri
yline(Limit_Inductive, 'r--', 'Endüktif Sınır (%20)', 'LineWidth', 1.5);
yline(-Limit_Capacitive, 'r--', 'Kapasitif Sınır (%15)', 'LineWidth', 1.5);
yline(0, 'k-', 'Hedef (0.00)', 'LineWidth', 1);

% 3. Sonuç Noktası
plot([0 1], [Y_Plot_Val Y_Plot_Val], 'b-', 'LineWidth', 2); % Seviye Çizgisi
scatter(0.5, Y_Plot_Val, 200, 'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);

% 4. Etiketleme
text(0.52, Y_Plot_Val, [' SONUÇ: %' num2str(CurrentRatio*100, '%.2f')], 'FontWeight', 'bold', 'FontSize', 10, 'BackgroundColor', 'white');

ylabel('Kapasitif (-) / Endüktif (+) Oran');
ylim([-0.5 0.5]); % Y ekseni aralığı (-%50 ile +%50 arası görünür)
set(gca, 'XTick', []); % X ekseni sayılarının önemi yok
legend('Endüktif Ceza', 'Kapasitif Ceza', 'Güvenli Bölge', 'Location', 'eastoutside');
hold off;

%% AMAÇ FONKSİYONU
function cost = CostFunction(x, steps, P, Q_load, lim_ind, lim_cap)
    Q_cap = sum(x .* steps);      
    Q_net = Q_load - Q_cap;       
    ratio = Q_net / P;            
    
    % Hata Maliyeti
    obj_error = abs(Q_net); 
    
    % Yasal Sınır Cezası (Çok Yüksek)
    penalty_legal = 0;
    if Q_net > 0 && (ratio > lim_ind)
        penalty_legal = 100000 * (ratio - lim_ind);
    elseif Q_net < 0 && (abs(ratio) > lim_cap)
        penalty_legal = 100000 * (abs(ratio) - lim_cap);
    end
    
    % Küçük Kademe Tercih Cezası (Ağırlıklı)
    % Büyük kondansatörlerin maliyetini artırıyoruz (steps.^1.5 veya steps.^2)
    preference_penalty = sum(x .* (steps .^ 1.5)) * 0.05;
    
    cost = obj_error + penalty_legal + preference_penalty;
end
