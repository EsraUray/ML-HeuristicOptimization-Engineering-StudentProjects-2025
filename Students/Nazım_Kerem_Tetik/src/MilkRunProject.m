clc; clear; close all;

%% 1. SİSTEM PARAMETRELERİ VE VERİ OLUŞTURMA
rng(42); 

n_stations = 10;      
n_vehicles = 2;       
Q_capacity = 20;      
EF = 0.8;             % Emisyon Faktörü (kg CO2 / km)

coords = [50, 50; randi([0, 100], n_stations, 2)];
depot = coords(1, :);
stations = coords(2:end, :);

demands = randi([2, 5], 1, n_stations);

dist_matrix = zeros(n_stations+1, n_stations+1);
for i = 1:n_stations+1
    for j = 1:n_stations+1
        dist_matrix(i,j) = sqrt(sum((coords(i,:) - coords(j,:)).^2));
    end
end

%% 2. GENETİK ALGORİTMA AYARLARI
pop_size = 50;        
num_iter = 200;       
mutation_rate = 0.1;  
elite_count = 2;      

population = zeros(pop_size, n_stations);
for i = 1:pop_size
    population(i, :) = randperm(n_stations);
end

%% 3. ALGORİTMA DÖNGÜSÜ
global_best_fitness = Inf;
global_best_route = [];
history = zeros(num_iter, 1);

fprintf('Genetik Algoritma Çalışıyor... Lütfen Bekleyin.\n');

for iter = 1:num_iter
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        [cost, ~, ~] = calculate_cost(population(i,:), demands, dist_matrix, Q_capacity, n_vehicles, EF);
        fitness(i) = cost;
        if cost < global_best_fitness
            global_best_fitness = cost;
            global_best_route = population(i,:);
        end
    end
    history(iter) = global_best_fitness;
    
    new_pop = zeros(size(population));
    [~, sort_idx] = sort(fitness);
    new_pop(1:elite_count, :) = population(sort_idx(1:elite_count), :);
    
    for k = elite_count+1:pop_size
        p1 = population(randi(pop_size), :);
        p2 = population(randi(pop_size), :);
        [c1, ~, ~] = calculate_cost(p1, demands, dist_matrix, Q_capacity, n_vehicles, EF);
        [c2, ~, ~] = calculate_cost(p2, demands, dist_matrix, Q_capacity, n_vehicles, EF);
        
        if c1 < c2, parent = p1; else, parent = p2; end
        
        child = parent;
        if rand < mutation_rate
            idx = randperm(n_stations, 2); 
            child([idx(1) idx(2)]) = child([idx(2) idx(1)]); 
        end
        new_pop(k, :) = child;
    end
    population = new_pop; 
end

%% 4. SONUÇLARI GÖRSELLEŞTİRME VE RAPORLAMA
[final_cost, tours, total_dist] = calculate_cost(global_best_route, demands, dist_matrix, Q_capacity, n_vehicles, EF);

% Final_cost KG cinsinden geliyor, Grama çevirelim
final_cost_gram = final_cost * 1000; 

fprintf('\n=== OPTİMİZASYON TAMAMLANDI ===\n');
fprintf('En İyi Toplam Mesafe: %.2f metre\n', total_dist);
fprintf('En İyi Toplam Emisyon: %.2f gram CO2 (%.4f kg)\n', final_cost_gram, final_cost);
fprintf('Kullanılan Araç Sayısı: %d\n', length(tours));

% --- Grafik 1: Yakınsama Grafiği (GRAM GÖSTERİMLİ) ---
figure('Name', 'Algoritma Performansı', 'Color', 'w');
% History verisini de grama çevirerek çizelim
plot(history * 1000, 'LineWidth', 2, 'Color', 'b'); 
title('Genetik Algoritma Yakınsama Grafiği', 'Color', 'k');
xlabel('İterasyon Sayısı', 'Color', 'k');
ylabel('Toplam Emisyon (gram CO2)', 'Color', 'k'); % Etiketi düzelttik
grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k');

% --- Grafik 2: En İyi Rota Haritası (GRAM GÖSTERİMLİ) ---
figure('Name', 'Optimum Rota', 'Color', 'w');
hold on;
scatter(coords(2:end,1), coords(2:end,2), 100, 'b', 'filled'); 
plot(depot(1), depot(2), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

colors = ['k', 'r', 'b', 'g']; 
legend_str = {'İstasyonlar', 'Depo'};

for t = 1:length(tours)
    vehicle_route = [0, tours{t}, 0]; 
    route_x = coords(vehicle_route+1, 1);
    route_y = coords(vehicle_route+1, 2);
    
    plot(route_x, route_y, 'LineWidth', 2.5, 'Color', colors(t), 'LineStyle', '-');
    legend_str{end+1} = sprintf('Araç %d (Yük: %d)', t, sum(demands(tours{t})));
end

for i = 1:n_stations
    text(coords(i+1,1)+1, coords(i+1,2)+1, sprintf('S%d(%d)', i, demands(i)), 'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold');
end
text(depot(1), depot(2)-2, 'DEPO', 'FontWeight', 'bold', 'Color', 'r');

% Başlığı GRAM cinsinden yazıyoruz
title(sprintf('Milk-Run Rotaları (Mesafe: %.1fm, CO2: %.1f gram)', total_dist, final_cost_gram), 'Color', 'k');
xlabel('X Koordinatı (m)', 'Color', 'k');
ylabel('Y Koordinatı (m)', 'Color', 'k');

lgd = legend(legend_str, 'Location', 'bestoutside');
set(lgd, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', 'k');

grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k');
hold off;


%% YARDIMCI FONKSİYON
function [total_emission, vehicle_tours, total_km] = calculate_cost(route_perm, demands, dists, Q, max_veh, EF)
    vehicle_tours = {};
    current_tour = [];
    current_load = 0;
    
    for i = 1:length(route_perm)
        station_idx = route_perm(i);
        load = demands(station_idx);
        
        if current_load + load <= Q
            current_tour = [current_tour, station_idx];
            current_load = current_load + load;
        else
            if ~isempty(current_tour)
                vehicle_tours{end+1} = current_tour;
            end
            current_tour = [station_idx];
            current_load = load;
        end
    end
    if ~isempty(current_tour)
        vehicle_tours{end+1} = current_tour;
    end
    
    total_km = 0;
    used_vehicles = length(vehicle_tours);
    
    for k = 1:used_vehicles
        tour = vehicle_tours{k};
        d = dists(1, tour(1)+1); 
        for j = 1:length(tour)-1
            d = d + dists(tour(j)+1, tour(j+1)+1);
        end
        d = d + dists(tour(end)+1, 1);
        total_km = total_km + d;
    end
    
    % HESAPLAMA KG CİNSİNDEN (DOĞRUSU BU)
    total_emission = (total_km / 1000) * EF; 
    
    if used_vehicles > max_veh
        total_emission = total_emission + 10000 * (used_vehicles - max_veh);
    end
end
