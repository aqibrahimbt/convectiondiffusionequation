#ifndef FILE_MYFESPACE_HPP
#define FILE_MYFESPACE_HPP

namespace ngcomp
{

// MyFESPACE is a derived class from FESpace
  class MyFESpace : public FESpace
  {
    bool secondorder;
    int ndof{}, nvert{};

    //BitArray
    shared_ptr<BitArray> my_ba;
    Array<int> gvert_to_cvert;

  public:
    /*
      constructor.
      Arguments are the access to the mesh data structure,
      and the flags from the define command in the pde-file
      or the kwargs in the Python constructor.
    */
    MyFESpace (shared_ptr<MeshAccess> ama, const Flags & flags, shared_ptr<BitArray> ba);

    /* a name for our new fe-space */
    virtual string GetClassName () const { return "MyFESpace"; }

    static DocInfo GetDocu();

    virtual void Update();

    // Get number of degrees of freedom
    virtual size_t GetNDof () const { return ndof; }

    virtual void GetDofNrs (ElementId ei, Array<DofId> & dnums) const;

    virtual FiniteElement & GetFE (ElementId ei, Allocator & alloc) const;

    // some new functionality our space should have in Python
    int GetNVert() { return nvert; }
  };

}

void ExportMyFESpace(py::module m);


#endif
