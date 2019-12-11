#include <comp.hpp>
#include <h1lofe.hpp>
#include <regex>
#include <python_comp.hpp>
#include <utility>
#include "myElement.hpp"
#include "myFESpace.hpp"
#include "myDiffOp.hpp"

// allows us to use the ngcomp namespace
namespace ngcomp
{

  MyFESpace :: MyFESpace (shared_ptr<MeshAccess> ama, const Flags & flags,
          shared_ptr<BitArray> ba)
    : FESpace (std::move(ama), flags), my_ba (std::move(ba))
  {
    cout << "Constructor of MyFESpace" << endl;
    cout << "Flags = " << flags << endl;
    cout << my_ba << endl;

    type = "myfespace";

    secondorder = flags.GetDefineFlag ("secondorder");
    cout << secondorder << endl;

    if (!secondorder)
      cout << "Working with first order elements" << endl;
    else
      throw Exception("no second order for now");

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

  // Updating Finite Element Space
  void MyFESpace :: Update()
  {
    // some global update:
    // GetNEdges: number of edges in the whole mesh
    // GetNV: number of vertices
    cout << "Update MyFESpace, #number of vertices = " << ma->GetNV()
         << ", #number of edges = " << ma->GetNEdges() << endl;

    // Get active vertices
    BitArray active_vertices(ma->GetNV());
    cout << "active vertices" << active_vertices << endl;
    active_vertices.Clear();

    cout << "range" << Range(ma->GetNE(VOL)) <<endl;

    // number of volume or boundary elements (GETNE)
    // loop over all the elements in the mesh
    for (auto i : Range(ma->GetNE(VOL)))
    {
        // check bit i
        if (my_ba->Test(i))
        {
            auto vs = ma->GetElVertices(ElementId(VOL,i));
            for (auto v : vs)
                active_vertices.Set(v);
        }
    }

    cout << "active vertices new" << active_vertices << endl;
    cout << "active vertices size" << active_vertices.Size() << endl;
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
  }

  void MyFESpace :: GetDofNrs (ElementId ei, Array<DofId> & dnums) const
  {
    // returns dofs of element ei
    // may be a volume triangle or boundary segment

    dnums.SetSize(0);

    if (!ei.IsVolume())
        return;
    if (!my_ba->Test(ei.Nr()))
        return;

    // first dofs are vertex numbers:
    for (auto v : ma->GetElVertices(ei))
      dnums.Append (gvert_to_cvert[v]);

  }

  FiniteElement & MyFESpace :: GetFE (ElementId ei, Allocator & alloc) const
  {

    if (!my_ba->Test(ei.Nr()))
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
  }
}

void ExportMyFESpace(py::module m)
{
  using namespace ngcomp;
    m.def("CustomFESpace", [](shared_ptr<MeshAccess> ma,
            shared_ptr<BitArray> ba, py::dict bpflags)-> shared_ptr<FESpace>
          {
              Flags flags = py::extract<Flags> (std::move(bpflags))();
              shared_ptr<FESpace> ret = make_shared<MyFESpace> (ma, flags, ba);
              ret->Update();
              ret->FinalizeUpdate();
              return ret;
          },
          docu_string(R"raw_string(...)raw_string"));

}
