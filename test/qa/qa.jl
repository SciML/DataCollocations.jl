using SciMLTesting, DataCollocations, Test
using JET

run_qa(
    DataCollocations;
    ei_kwargs = (;
        # `fast_scalar_indexing` is not (yet) public in ArrayInterface.
        all_explicit_imports_are_public = (; ignore = (:fast_scalar_indexing,)),
    ),
)
