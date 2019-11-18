clear;
warning('off');
close all; clc
addpath './tool';
addpath './data';
addpath './metric';
noise = [0.03, 0.05, 0.07, 0.1];
shape_files = [600, 1200, 1200, 1500];
%% todo
for shape = 1:1
    switch shape
        case 1
            shapename = 'soldier';
        case 2
            shapename = 'longdress';
        case 3
            shapename = 'loot';
        case 4
            shapename = 'redandblack';
    end
    for noisetype = 4:4
        
        
        for file_e = 1:6
            file = file_e + shape_files(shape);
            %shapenam e = 'gargoyle';%dc,gargoyle,anchor,daratech,lordquas
            gt_filename = [shapename , '_vox10_',num2str(file, '%04d'), 'ds.ply'];
            % noisename = [shapename,'_gaussian_noise_',num2str(0.03)];%0.03,0.04
            n_filename = [shapename , '_vox10_',num2str(file, '%04d'), 'ds_gaussian_noise_', num2str(noise(noisetype)), '.ply'];
            n_filename_ne = [shapename , '_vox10_',num2str(file, '%04d'), 'ds_gaussian_noise_', num2str(noise(noisetype))];
            if file_e == 1
                np_filename = gt_filename; %empty file
            else
                np_filename = [shapename , '_vox10_',num2str(file-1, '%04d'), 'ds_gaussian_noise_', num2str(noise(noisetype)),'_dn.ply'];% pre frame
            end
            % nn_filename = ['soldier_vox10_0601_ds_gaussian_noise_0.05.ply']; % next frame
            
            %% read in file for ground-truth and noisy model
            pt_gt = pcread(gt_filename);
            X_gt = pt_gt.Location;
            
            size_max = max(max(X_gt)) / 60;
            
            pt_X= pcread(n_filename);
            X = pt_X.Location;
            pt_p = pcread(np_filename);
            X_p = pt_p.Location;
            % scale normalize
            X_gt = X_gt / size_max;
            pt_gt = pointCloud(X_gt, 'Color', pt_gt.Color);
            X = X/size_max;
            pt_X = pointCloud(X, 'Color', pt_X.Color);
            X_p = X_p / size_max;
            pt_p = pointCloud(X_p, 'Color', pt_p.Color);
            temp = meandistance(X_gt * size_max, X * size_max);
            disp(temp);
            %% todo end
            
            %% parameter settings
            max_itr = 4; change_tolerance = 0.0001; % stop criteria
            % choose fps for uniform sampling
            flag_sample = 1; SAMPLING_SET = ceil(0.5*size(X,1));
            % if not, use grid sample
            gridStep = 0.7;
            % patch size, searching window size,
            pk = 20; wink = 9;
            % graph construction parameter: epsilon, patch similarity threshold
            eps1 = 0.45*ones(max_itr,1); eps1(1:2)=[1.05, 0.75]; eps1 = eps1.^2;
            threshold_d = 20;
            % optimization parameter
            gamma=0.5; lambda1 = 0.2*ones(max_itr,1); lambda2=zeros(max_itr, 1);
            lambda1(1:4) = [3 25 70 300];
            if file_e ~= 1
                lambda2(1:2) = [0.4, 0];
            end
            %% todo
            winf = 8; % window to find most similar patch in adjcent frames
            %% todo end
            
            inter_normal = pcnormals(pt_p, pk);
            
            %% pixel graph (no prefiltering now)
            X_m = X;
            
            %% patch center, patch construction with size=pk, and patch graph with k=wink
            pt_pre = pointCloud(X);
            % patch center
            if flag_sample
                srf = struct('X',X(:,1),'Y',X(:,2),'Z',X(:,3));
                ifps = fps_euc(srf,SAMPLING_SET);
                pt_C = pointCloud(X(ifps,:));
            else
                pt_C = pcdownsample(pt_pre,'gridAverage',gridStep);
            end
            % patch construction
            pn = pt_C.Count; % patch number
            P = zeros(pn,pk); % patch center and the node indices in the patch
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_pre,pt_C.Location(i,:),pk);
                P(i,:) = indices;
            end
            Nf = pn*pk; % node graph size
            % patch graph
            P_win = zeros(pn, wink);
            P_winf = zeros(pn, winf);
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_C,pt_C.Location(i,:),wink);
                P_win(i,:) = indices;
                
            end
            N_p = pt_p.Count; % pre frame point number
            P_p_win = zeros(N_p, pk);
            for i = 1:N_p
                [indices,dists] = findNearestNeighbors(pt_p,pt_p.Location(i,:),pk);
                P_p_win(i,:) = indices;
            end
            P_win = P_win(:,2:end);
            
            %% Denoising
            dM = zeros(max_itr,1); dH = dM;
            X_pre = X;
            % set center
            X_c = zeros(pn,3);
            for c = 1:3
                u = X_pre(:,c);
                X_c(:,c) = mean(u(P),2);
            end
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_p,X_c(i,:),winf);
                P_winf(i,:) = indices;
            end
            
            for itr = 1:max_itr
                %% prepare reference frame for each patch: so as to compute height field and projection
                % shift the point wrt the center
                
                pt_pre = pointCloud(X_pre);
                intra_normal = pcnormals(pt_pre, pk);
                
                inter_F = [X_p inter_normal]; % n*6
                intra_F = [X_pre intra_normal]; % n*6
                f = zeros(pn,pk,3); fo = f;
                for c = 1:3
                    u = X_m(:,c);
                    f(:,:,c) = u(P)-repmat(X_c(:,c),1,pk);
                    u = X_pre(:,c);
                    fo(:,:,c) = u(P)-repmat(X_c(:,c),1,pk);
                end
                % normal at each center
                fn = zeros(pn,3); ptmp = zeros(pk,3);
                for i = 1:pn
                    ptmp(:,:) = f(i,:,:);
                    cov = ptmp'*ptmp;
                    [Vtmp,Dtmp] = eig(cov);
                    fn(i,:) = Vtmp(:,1); % reference plane with normal vertor stored in fn
                end
                %% todo
                P_p = zeros(pk*pn, 3);
                P_n = zeros(pk*pn, 3);
                %% end todo
                
                %% patch similarity and connection
                D_i = zeros(Nf,wink-1);D_w = zeros(Nf,wink-1); tmp_p1 = (1:pk)';
                D_wp = zeros(Nf, 1);
                % temp variable for comparing fixed and moving patches
                fixed = zeros(pk,3); fixed_n = zeros(3,1); moving = fixed;
                
                
                patch_S = zeros(pn * (wink - 1) * pk, 6);
                patch_T = zeros(pn * (wink - 1) * pk, 6);
                % record patches for corresponding patches inter frame
                patch_S_ = zeros(pn * pk, 6);
                patch_T_ = zeros(pn * pk, 6);
                for i = 1:pn
                    fixed(:,:) = f(i,:,:); fixed_n(:) = fn(i,:);
                    % compute height field, and projection on reference plane
                    hf = fixed*fixed_n; fixed_proj = fixed-hf*fixed_n';
                    % search the neighbors of fixed
                    wedge = P_win(i,:);
                    for j = 1:length(wedge)
                        % load moving
                        w = wedge(j); moving(:,:) = f(w,:,:);
                        % height field and projection on plane
                        hm = moving*fixed_n; moving_proj = moving-hm*fixed_n';
                        % use projection for neighbor search
                        xx = sum((fixed_proj.^2),2);
                        yy = sum((moving_proj.^2),2);
                        xy = fixed_proj*moving_proj';
                        DistMat = repmat(xx,1,length(yy))-2*xy+repmat(yy',length(xx),1);
                        [dmin, idx] = min(DistMat,[],2);
                        % use height field for computing distance
                        for k = 1:pk
                            patch_S(pk*(i - 1)*(wink - 1)+(k - 1) * (wink - 1) + j,:) = intra_F(P(i,k),:);
                            patch_T(pk*(i - 1)*(wink - 1)+(k - 1) * (wink - 1) + j,:) = intra_F(P(w,idx(k)),:);
                        end
                        D_i(pk*(i-1)+tmp_p1,j) = pk*(w-1)+idx(:);
                        %% todo end
                    end
                    %% todo
                    % find most similar patch in pre frame
                    pt_p = pointCloud(X_p);
                    indices = P_winf(i,:);
                    Dist_min = 999999;
                    for j = 1:length(indices)
                        % build patch
                        indices_p = P_p_win(indices(j),:);
                        center = zeros(1,3);
                        for c=1:3
                            center(c) = mean(X_p(indices_p,c));
                        end
                        moving = X_p(indices_p,:) - repmat(center, pk, 1);
                        % height field and projection on plane
                        hm = moving*fixed_n; moving_proj = moving-hm*fixed_n';
                        % use projection for neighbor search
                        xx = sum((fixed_proj.^2),2);
                        yy = sum((moving_proj.^2),2);
                        xy = fixed_proj*moving_proj';
                        DistMat = repmat(xx,1,length(yy))-2*xy+repmat(yy',length(xx),1);
                        [dmin, idx] = min(DistMat,[],2);
                        % use height field for computing distance
                        Dist = (sum(dmin))/pk;
                        if Dist < Dist_min
                            Dist_min = Dist;
%                             D_wp(pk*(i-1)+tmp_p1) = Dist;
                            patch_S_((i - 1) * pk + 1: i * pk, :) = intra_F(P(i,:),:);
                            patch_T_((i - 1) * pk + 1: i * pk, :) = inter_F(indices_p(idx),:);
                            P_p(pk*(i-1)+tmp_p1, :) = moving(idx(:), :);
                        end
                    end
                    
                end
                
                [R,D_w, pr] = proximal_gradient_descent16(patch_S, patch_T, true); % use normal
                R
                pr'
                D_w = reshape(D_w, wink - 1, pn * pk)';
                histogram(D_w);
                [R_,D_wp, pr] = proximal_gradient_descent16(patch_S_, patch_T_, true); % use normal
                pr'
                
                tmp = repmat((1:Nf)',1,wink-1);
                D_p = sparse(tmp(:),D_i(:),D_w(:),Nf,Nf);
                D_pp = sparse(tmp(:,1), tmp(:,1), D_wp(:), Nf, Nf);
%                 D_pp(D_pp>0) = exp(-D_pp(D_pp>0)./(2*eps1(itr)));
                A = D_p; A(A>20) = 0;
                mask1 = logical(tril(A));
                A = A-mask1'.*A;
                A = A+A';
%                 A(A>0)=exp(-A(A>0)./(2*eps1(itr)));
                weight_rec=diag(sum(A).^(-gamma));
                A=weight_rec*A*weight_rec;
                Dn=sum(A);
                Dn=diag(Dn);
                L=Dn-A;
                
                X_rec = zeros(length(X_pre),3);
                tmp = P';
                U = sparse(1:Nf,tmp(:),ones(Nf,1),Nf,length(X_pre));
                
           
                for c = 1:3
                    t = repmat(X_c(:,c),1,pk);
                    t = t'; t = t(:);
                    u0 = X_pre(:,c);
%                     
                    C = U'*L*t+lambda1(itr)*u0+lambda2(itr)*U'*D_pp*(t+P_p(:,c));
                    B = U'*L*U+lambda1(itr)*speye(length(X_pre))+lambda2(itr)*U'*D_pp*U;
                    X_rec(:,c) = lsqr(B,double(C),1e-06,10000);
                    
                end
                pc = pointCloud(X_rec*size_max, 'Color', pt_X.Color);
                name = ['./data/' n_filename_ne 'xrec_' num2str(itr) '_' num2str(lambda1(itr)) '_' num2str(lambda2(itr)) '.ply'];
                pcwrite(pc, name);
%                 write_ply_only_points(X_rec,['./data/' n_filename_ne 'xrec_' num2str(itr) '_' num2str(lambda1(itr)) '_' num2str(lambda2(itr)) '.ply']);
                dM(itr,1) = meandistance(X_gt*size_max, X_rec*size_max);
                change_dM = meandistance(X_pre*size_max, X_rec*size_max);
                disp([num2str(itr), ',', num2str(lambda1(itr)), ',', num2str(lambda2(itr)), ',' ,num2str(dM(itr,1)) ', ' num2str(change_dM) '\n']);
                % 1 2.1852
                if change_dM < change_tolerance % terminate if it doesn't change too much
                    break;
                end
                
                
                %     end
                X_pre = X_rec;
                X_m = X_pre;
            end
            pcwrite(pc, ['./data/' n_filename_ne '_dn.ply']);
            disp(['meandistance=',num2str(mdc)]);
        end
    end
end
