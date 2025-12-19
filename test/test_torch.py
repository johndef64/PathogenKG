"""
Test script for PyTorch with CUDA, PyTorch Geometric, and torch-sparse
"""

import sys
import torch
import platform

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_pytorch_cuda():
    """Test PyTorch installation and CUDA availability"""
    print_section("PyTorch and CUDA Information")
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
        
        # Test basic CUDA operations
        print_section("Testing Basic CUDA Operations")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print("✓ Matrix multiplication on CUDA: SUCCESS")
            print(f"  Result shape: {z.shape}")
            print(f"  Result device: {z.device}")
            
            # Test memory allocation
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  Memory allocated: {allocated:.3f} GB")
            print(f"  Memory reserved: {reserved:.3f} GB")
            
            del x, y, z
            torch.cuda.empty_cache()
            print("✓ Memory cleanup: SUCCESS")
            
        except Exception as e:
            print(f"✗ CUDA operations failed: {e}")
    else:
        print("⚠ CUDA is not available. Running on CPU only.")

def test_torch_geometric():
    """Test PyTorch Geometric installation"""
    print_section("PyTorch Geometric Information")
    
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
        
        # Test basic PyG functionality
        from torch_geometric.data import Data
        
        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            edge_index = edge_index.to(device)
            x = x.to(device)
            
        data = Data(x=x, edge_index=edge_index)
        print("✓ PyTorch Geometric Data creation: SUCCESS")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Number of edges: {data.num_edges}")
        print(f"  Device: {data.x.device}")
        
        # Test a simple GNN layer
        from torch_geometric.nn import GCNConv
        
        conv = GCNConv(16, 32)
        if torch.cuda.is_available():
            conv = conv.to(device)
        
        out = conv(data.x, data.edge_index)
        print("✓ GCN layer forward pass: SUCCESS")
        print(f"  Output shape: {out.shape}")
        print(f"  Output device: {out.device}")
        
    except ImportError as e:
        print(f"✗ PyTorch Geometric not installed: {e}")
    except Exception as e:
        print(f"✗ PyTorch Geometric test failed: {e}")

def test_torch_sparse():
    """Test torch-sparse installation and operations"""
    print_section("Torch Sparse Information")
    
    try:
        import torch_sparse
        print(f"torch-sparse version: {torch_sparse.__version__}")
        
        # Test sparse matrix operations
        row = torch.tensor([0, 1, 1, 2, 2, 2])
        col = torch.tensor([1, 0, 2, 0, 1, 2])
        value = torch.randn(6)
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            row = row.to(device)
            col = col.to(device)
            value = value.to(device)
        
        # Create sparse tensor
        from torch_sparse import SparseTensor
        
        sparse_tensor = SparseTensor(row=row, col=col, value=value, 
                                     sparse_sizes=(3, 3))
        print("✓ SparseTensor creation: SUCCESS")
        print(f"  Shape: {sparse_tensor.sizes()}")
        print(f"  Number of non-zero elements: {sparse_tensor.nnz()}")
        print(f"  Device: {value.device}")
        
        # Test sparse matrix multiplication
        x = torch.randn(3, 16)
        if torch.cuda.is_available():
            x = x.to(device)
        
        out = sparse_tensor @ x
        print("✓ Sparse matrix multiplication: SUCCESS")
        print(f"  Output shape: {out.shape}")
        print(f"  Output device: {out.device}")
        
    except ImportError as e:
        print(f"✗ torch-sparse not installed: {e}")
    except Exception as e:
        print(f"✗ torch-sparse test failed: {e}")

def test_additional_packages():
    """Test additional PyG extension packages"""
    print_section("Additional PyG Extension Packages")
    
    packages = [
        ('torch_scatter', 'Scatter operations'),
        ('torch_cluster', 'Clustering operations'),
        ('torch_spline_conv', 'Spline convolutions')
    ]
    
    for package_name, description in packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name} ({description}): v{version}")
        except ImportError:
            print(f"✗ {package_name} ({description}): NOT INSTALLED")

def main():
    """Run all tests"""
    print("\n" + "🔥"*30)
    print("PyTorch CUDA Test Suite")
    print("🔥"*30)
    
    try:
        test_pytorch_cuda()
        test_torch_geometric()
        test_torch_sparse()
        test_additional_packages()
        
        print_section("Test Summary")
        if torch.cuda.is_available():
            print("✓ All CUDA tests completed successfully!")
            print(f"Default CUDA device: {torch.cuda.current_device()}")
        else:
            print("⚠ Tests completed but CUDA is not available")
            
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()

