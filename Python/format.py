 ipintegrator = SymbolicBFI(
      u() * v(), bonus_intorder=bonus_int)

  ba_active_elements = BitArray(mesh.ne)

   for enr_indicator in self.config['enrich_domain_ind']:
        ba_active_elements |= mark_elements(
            mesh, enr_indicator, size)

    for el in fes.Elements():
        if ba_active_elements[el.nr]:
            i = ElementId(el)
            N = len(el.dofs)
            element = fes.GetFE(el)
            elementstd = V.GetFE(i)
            Nstd = elementstd.ndof
            trafo = mesh.GetTrafo(i)
            # Get element matrix
            elmat = ipintegrator.CalcElementMatrix(element, trafo)
            # print(elmat)
            # input('')
            important = [True if el.dofs[i] >= 0 else False for i in range(N)]
            try:
                factors = []
                for i in range(Nstd, N):
                    if important[i]:
                        active = [j for j in range(i) if important[j]]
                        #print(elmat[i, i])
                        # input('test')
                        # Division by zero
                        factor = 1 - 2 * \
                            sum([elmat[i, j] ** 2/elmat[i, i] /
                                 elmat[j, j] for j in active])
                        factor += sum([elmat[i, j]*elmat[i, k]*elmat[j, k]/elmat[i, i] /
                                       elmat[j, j]/elmat[k, k] for j in active for k in active])
                        factor = sqrt(abs(factor))

                        factors.append(factor)
                        if (factor <= 1e-3):
                            print('yes')
                            important[i] = False
                            if el.dofs[i] >= 0:
                                ba_active_dofs[el.dofs[i]
                                               ] = False
