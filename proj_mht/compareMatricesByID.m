function result = compareMatricesByID(M1, rowIDs1, colIDs1, ...
                                      M2, rowIDs2, colIDs2)
%COMPAREMATRICESBYID  Align and compare two matrices based on row/column IDs.
%
%   result = compareMatricesByID(M1, rowIDs1, colIDs1, ...
%                                M2, rowIDs2, colIDs2)
%
%   Returns a struct containing:
%       .A1         M1 aligned and sorted by ID
%       .A2         M2 aligned and sorted by ID
%       .rowIDs     aligned row IDs
%       .colIDs     aligned column IDs
%       .diff       element-wise difference A1 - A2
%       .equal      logical matrix (true where A1 == A2)
%
%   Notes:
%     - Rows/columns appearing in only one matrix are removed.
%     - IDs must be numeric.

    % ---- Sort rows/columns of each input matrix independently ----
    [r1_sorted, rmap1] = sort(rowIDs1);
    [c1_sorted, cmap1] = sort(colIDs1);
    M1s = M1(rmap1, cmap1);

    [r2_sorted, rmap2] = sort(rowIDs2);
    [c2_sorted, cmap2] = sort(colIDs2);
    M2s = M2(rmap2, cmap2);

    % ---- Find common row IDs and map into both matrices ----
    [rowIDs, ia1, ia2] = intersect(r1_sorted, r2_sorted);

    % ---- Find common column IDs ----
    [colIDs, ib1, ib2] = intersect(c1_sorted, c2_sorted);

    % ---- Extract aligned submatrices ----
    A1 = M1s(ia1, ib1);
    A2 = M2s(ia2, ib2);

    % ---- Produce comparison results ----
    diffMatrix = A1 - A2;
    equalMatrix = A1 == A2;

    % ---- Output struct ----
    result = struct();
    result.A1     = A1;
    result.A2     = A2;
    result.rowIDs = rowIDs;
    result.colIDs = colIDs;
    result.diff   = diffMatrix;
    result.equal  = equalMatrix;
end
