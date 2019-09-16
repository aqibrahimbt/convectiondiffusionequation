/*********************************************************************/
/* File:   myFESpace.cpp                                             */
/* Author: Joachim Schoeberl                                         */
/* Date:   26. Apr. 2009                                             */
/*********************************************************************/


/*

My own FESpace for linear and quadratic triangular elements.

A fe-space provides the connection between the local reference
element, and the global mesh.

*/


#include <comp.hpp>    // provides FESpace, ...
#include <h1lofe.hpp>
#include <regex>
#include <python_comp.hpp>
#include "myElement.hpp"
#include "myFESpace.hpp"
#include "myDiffOp.hpp"


namespace ngcomp
{

  MyFESpace :: MyFESpace (shared_ptr<MeshAccess> ama, const Flags & flags, shared_ptr<BitArray> ba)
    : FESpace (ama, flags), my_ba (ba)
  {
    cout << "Constructor of MyFESpace" << endl;
    cout << "Flags = " << flags << endl;
    cout << my_ba << endl;
    // this is needed for pickling and needs to be the same as used in
    // RegisterFESpace later
    type = "myfespace";

    secondorder = flags.GetDefineFlag ("secondorder");

    if (!secondorder)
      cout << "You have chosen first order elements" << endl;
    else
        throw Exception("no second order for now");
    //cout << "You have chosen second order elements" << endl;

    // needed for symbolic integrators and to draw solution
    evaluator[VOL] = make_shared<T_DifferentialOperator<MyDiffOpId>>();
    flux_evaluator[VOL] = make_shared<T_DifferentialOperator<MyDiffOpGradient>>();
  }

  DocInfo MyFESpace :: GetDocu()
  {
    auto docu = FESpace::GetDocu();
    docu.Arg("secondorder") = "bool = False\n"
      "  Use second order basis functions";
    return docu;
  }

  void MyFESpace :: Update()
  {
    // some global update:
    cout << "Update MyFESpace, #vert = " << ma->GetNV()
         << ", #edge = " << ma->GetNEdges() << endl;

    BitArray active_vertices(ma->GetNV());
    active_vertices.Clear();

    for (auto i : Range(ma->GetNE(VOL)))
    {
        if (my_ba->Test(i))
        {
            auto vs = ma->GetElVertices(ElementId(VOL,i));
            for (auto v : vs)
                active_vertices.Set(v);
        }
    }
    cout << active_vertices << endl;
    ndof = 0;
    gvert_to_cvert.SetSize(ma->GetNV());

    for (int i = 0; i < active_vertices.Size(); i++)
        if (active_vertices.Test(i))
        {
            gvert_to_cvert[i] = ndof;
            ndof++;
        }
        else
            gvert_to_cvert[i] = -1;
    cout << "ndof: " << ndof << endl;
    // numbe   r of vertices
    //nvert = ma->GetNV();

    // number of dofs:
    //ndof = nvert;
    //if (secondorder)
    //  ndof += ma->GetNEdges();  // num vertics + num edges
  }

  void MyFESpace :: GetDofNrs (ElementId ei, Array<DofId> & dnums) const
  {
    // returns dofs of element ei
    // may be a volume triangle or boundary segment

    dnums.SetSize(0);

    if (!ei.IsVolume())
        return;
    if (! my_ba->Test(ei.Nr()))
        return;

    // first dofs are vertex numbers:
    for (auto v : ma->GetElVertices(ei))
      dnums.Append (gvert_to_cvert[v]);

    if (secondorder)
      {
        // more dofs on edges:
        for (auto e : ma->GetElEdges(ei))
          dnums.Append (nvert+e);
      }
    cout << ei << dnums << endl;
  }

  FiniteElement & MyFESpace :: GetFE (ElementId ei, Allocator & alloc) const
  {

    if (! my_ba->Test(ei.Nr()))
        return * new (alloc) DummyFE<ET_TRIG>;

    if (ei.IsVolume())
      {
        if (!secondorder)
          return * new (alloc) MyLinearTrig;
        else
          return * new (alloc) MyQuadraticTrig;
      }
    else
      throw Exception("Boundary elements not implemented yet!");
    // else
    //   {
    //     if (!secondorder)
    //       return * new (alloc) MyLinearSegm;
    //     else
    //       return * new (alloc) MyQuadraticSegm;
    //   }
  }

  /*
    register fe-spaces
    Object of type MyFESpace can be defined in the pde-file via
    "define fespace v -type=myfespace"
  */

  //static RegisterFESpace<MyFESpace> initifes ("myfespace");
}

void ExportMyFESpace(py::module m)
{
  using namespace ngcomp;
    m.def("MyFESpace2", [](shared_ptr<MeshAccess> ma, shared_ptr<BitArray> ba, py::dict bpflags)
                  -> shared_ptr<FESpace>
          {
              Flags flags = py::extract<Flags> (bpflags)();
              shared_ptr<FESpace> ret = make_shared<MyFESpace> (ma, flags, ba);
              //LocalHeap lh (1000000, "SFESpace::Update-heap", true);
              ret->Update();
              ret->FinalizeUpdate();
              return ret;
          },
          docu_string(R"raw_string(
...
)raw_string"));


  //ExportFESpace<MyFESpace>(m, "MyFESpace")
  //  .def("GetNVert", &MyFESpace::GetNVert)
  //  ;
}
